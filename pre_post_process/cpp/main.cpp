#include "stru_parser.h"
#include "csr_parser.h"
#include "block_parser.h"
#include "neighbor_list.h"
#include "transformer.h"
#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace {

void write_tensor_raw(const fs::path& path, const torch::Tensor& tensor) {
    torch::Tensor contiguous = tensor.contiguous().cpu();
    std::ofstream fout(path, std::ios::binary);
    if (!fout) {
        throw std::runtime_error("failed to open " + path.string() + " for writing");
    }
    fout.write(
        reinterpret_cast<const char*>(contiguous.data_ptr()),
        contiguous.numel() * contiguous.element_size()
    );
    if (!fout) {
        throw std::runtime_error("failed to write " + path.string());
    }
}

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 2);
    for (char ch : value) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += ch; break;
        }
    }
    return out;
}

void save_nxraw(const fs::path& output_dir,
                const TransformationEngine::ProcessedData& processed,
                const StruData& stru,
                int nspin,
                double cutoff) {
    fs::create_directories(output_dir);

    write_tensor_raw(output_dir / "descriptor.bin", processed.descriptor);
    write_tensor_raw(output_dir / "mask.bin", processed.mask);
    write_tensor_raw(output_dir / "edge_vec.bin", processed.edge_vec);
    write_tensor_raw(output_dir / "edge_src.bin", processed.edge_src);
    write_tensor_raw(output_dir / "edge_dst.bin", processed.edge_dst);

    const auto num_edges = processed.edge_src.size(0);
    const auto orbital_dim = processed.descriptor.size(1);
    std::ofstream manifest(output_dir / "manifest.json");
    if (!manifest) {
        throw std::runtime_error("failed to open nxraw manifest for writing");
    }

    manifest << "{\n";
    manifest << "  \"format\": \"nextham-nxraw\",\n";
    manifest << "  \"version\": 1,\n";
    manifest << "  \"nspin\": " << nspin << ",\n";
    manifest << "  \"cutoff\": " << cutoff << ",\n";
    manifest << "  \"num_edges\": " << num_edges << ",\n";
    manifest << "  \"num_atoms\": " << stru.atoms.size() << ",\n";
    manifest << "  \"orbital_dim\": " << orbital_dim << ",\n";
    manifest << "  \"output_path\": \"output_inference.pth\",\n";
    manifest << "  \"atom_elements\": [";
    for (size_t i = 0; i < stru.atoms.size(); ++i) {
        if (i != 0) manifest << ", ";
        manifest << "\"" << json_escape(stru.atoms[i].element) << "\"";
    }
    manifest << "],\n";
    manifest << "  \"tensors\": {\n";
    manifest << "    \"descriptor\": {\"file\": \"descriptor.bin\", \"dtype\": \"complex64\", \"shape\": ["
             << processed.descriptor.size(0) << ", " << processed.descriptor.size(1) << ", "
             << processed.descriptor.size(2) << "]},\n";
    manifest << "    \"mask\": {\"file\": \"mask.bin\", \"dtype\": \"float32\", \"shape\": ["
             << processed.mask.size(0) << ", " << processed.mask.size(1) << ", "
             << processed.mask.size(2) << "]},\n";
    manifest << "    \"edge_vec\": {\"file\": \"edge_vec.bin\", \"dtype\": \"float32\", \"shape\": ["
             << processed.edge_vec.size(0) << ", " << processed.edge_vec.size(1) << "]},\n";
    manifest << "    \"edge_src\": {\"file\": \"edge_src.bin\", \"dtype\": \"int64\", \"shape\": ["
             << processed.edge_src.size(0) << "]},\n";
    manifest << "    \"edge_dst\": {\"file\": \"edge_dst.bin\", \"dtype\": \"int64\", \"shape\": ["
             << processed.edge_dst.size(0) << "]}\n";
    manifest << "  }\n";
    manifest << "}\n";
}

void save_torch_pth(const std::string& output_pth,
                    const TransformationEngine::ProcessedData& processed) {
    // Match Python's [descriptor, overlap, mask, edge_vec, edge_src, edge_dst, ele_list, output_path_str]
    c10::impl::GenericList list(c10::AnyType::get());
    list.push_back(processed.descriptor);
    list.push_back(torch::IValue()); // Set overlap to None
    list.push_back(processed.mask);
    list.push_back(processed.edge_vec);
    list.push_back(processed.edge_src);
    list.push_back(processed.edge_dst);

    // ele_list (list of lists of strings, to be interpreted as list of tuples in Python)
    c10::impl::GenericList ele_ivalue_list(c10::AnyType::get());
    for (const auto& p : processed.ele_list) {
        c10::impl::GenericList tuple_list(c10::AnyType::get());
        tuple_list.push_back(p.first);
        tuple_list.push_back(p.second);
        ele_ivalue_list.push_back(tuple_list);
    }
    list.push_back(ele_ivalue_list);
    list.push_back(std::string("output_inference.pth"));

    // Match Python's (data_list, label) tuple
    std::vector<torch::IValue> tuple_elements;
    tuple_elements.push_back(list);
    tuple_elements.push_back(torch::IValue()); // label is None
    auto root_tuple = c10::ivalue::Tuple::create(std::move(tuple_elements));

    // Use pickle_save to match Python's torch.save
    auto bytes = torch::jit::pickle_save(torch::IValue(root_tuple));
    std::ofstream fout(output_pth, std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <structure_file> <data_dir> <nspin> <cutoff> <output_path>"
                  << " [--format stru|cif] [--output-format torch|nxraw]" << std::endl;
        return 1;
    }

    std::string stru_file = argv[1];
    std::string data_dir = argv[2];
    int nspin = std::stoi(argv[3]);
    double cutoff = std::stod(argv[4]);
    std::string output_path = argv[5];
    std::string structure_format = "stru";
    std::string output_format = "torch";
    for (int i = 6; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--format" && i + 1 < argc) {
            structure_format = argv[++i];
        } else if (arg == "--output-format" && i + 1 < argc) {
            output_format = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << std::endl;
            return 1;
        }
    }
    if (structure_format != "stru" && structure_format != "cif") {
        std::cerr << "Unsupported structure format: " << structure_format
                  << " (expected stru or cif)" << std::endl;
        return 1;
    }
    if (output_format != "torch" && output_format != "nxraw") {
        std::cerr << "Unsupported output format: " << output_format
                  << " (expected torch or nxraw)" << std::endl;
        return 1;
    }

    try {
        auto total_start = std::chrono::high_resolution_clock::now();

        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "Parsing " << structure_format << " structure..." << std::endl;
        auto stru_data = structure_format == "cif"
            ? StruParser::parse_cif(stru_file)
            : StruParser::parse(stru_file);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "  Structure parsing took: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

        TransformationEngine::ProcessedData processed;
        TransformationEngine engine(nspin);

        std::string csr_file = data_dir + "/hrs1_nao.csr";
        if (fs::exists(csr_file)) {
            start = std::chrono::high_resolution_clock::now();
            std::cout << "Parsing CSR..." << std::endl;
            auto csr_data = CSRParser::parse(csr_file, nspin);
            end = std::chrono::high_resolution_clock::now();
            std::cout << "  CSR parsing took: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

            start = std::chrono::high_resolution_clock::now();
            std::cout << "Building Neighbor List..." << std::endl;
            auto edges = NeighborList::build(stru_data, cutoff);
            end = std::chrono::high_resolution_clock::now();
            std::cout << "  Neighbor list build took: " << std::chrono::duration<double>(end - start).count() << " s (" << edges.size() << " edges found)" << std::endl;

            start = std::chrono::high_resolution_clock::now();
            std::cout << "Transforming data..." << std::endl;
            processed = engine.process(stru_data, csr_data, edges);
            end = std::chrono::high_resolution_clock::now();
            std::cout << "  Transformation took: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;
        } else {
            start = std::chrono::high_resolution_clock::now();
            std::cout << "Parsing Blocks (prefix hrs_block_up)..." << std::endl;
            auto block_data = BlockParser::parse_dir(data_dir, "hrs_block_up", nspin);
            end = std::chrono::high_resolution_clock::now();
            std::cout << "  Block parsing took: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

            start = std::chrono::high_resolution_clock::now();
            std::cout << "Building Neighbor List..." << std::endl;
            auto edges = NeighborList::build(stru_data, cutoff);
            end = std::chrono::high_resolution_clock::now();
            std::cout << "  Neighbor list build took: " << std::chrono::duration<double>(end - start).count() << " s (" << edges.size() << " edges found)" << std::endl;

            start = std::chrono::high_resolution_clock::now();
            std::cout << "Transforming data..." << std::endl;
            processed = engine.process(stru_data, block_data, edges);
            end = std::chrono::high_resolution_clock::now();
            std::cout << "  Transformation took: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;
        }

        start = std::chrono::high_resolution_clock::now();
        std::cout << "Saving to " << output_path << " (" << output_format << ")..." << std::endl;
        if (output_format == "nxraw") {
            save_nxraw(output_path, processed, stru_data, nspin, cutoff);
        } else {
            save_torch_pth(output_path, processed);
        }
        end = std::chrono::high_resolution_clock::now();
        std::cout << "  Saving took: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

        auto total_end = std::chrono::high_resolution_clock::now();
        std::cout << "Total Preprocessing time: " << std::chrono::duration<double>(total_end - total_start).count() << " s" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
