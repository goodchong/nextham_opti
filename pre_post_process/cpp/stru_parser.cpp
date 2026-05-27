#include "stru_parser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace {

std::string trim(const std::string& input) {
    const auto first = input.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = input.find_last_not_of(" \t\r\n");
    return input.substr(first, last - first + 1);
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return value;
}

std::vector<std::string> cif_tokens(const std::string& line) {
    std::vector<std::string> tokens;
    std::string token;
    char quote = '\0';
    for (char c : line) {
        if (quote != '\0') {
            if (c == quote) {
                tokens.push_back(token);
                token.clear();
                quote = '\0';
            } else {
                token += c;
            }
        } else if (c == '\'' || c == '"') {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
            quote = c;
        } else if (c == '#') {
            break;
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
        } else {
            token += c;
        }
    }
    if (!token.empty()) tokens.push_back(token);
    return tokens;
}

double cif_number(const std::string& value, const std::string& name) {
    if (value == "." || value == "?") {
        throw std::runtime_error("Missing CIF numeric value for " + name);
    }
    const auto uncertainty = value.find('(');
    return std::stod(value.substr(0, uncertainty));
}

std::string element_from_cif(const std::string& value) {
    std::string element;
    for (char c : value) {
        if (!std::isalpha(static_cast<unsigned char>(c))) break;
        element += c;
    }
    if (element.empty()) {
        throw std::runtime_error("Could not determine element from CIF atom value: " + value);
    }
    element[0] = std::toupper(static_cast<unsigned char>(element[0]));
    for (size_t i = 1; i < element.size(); ++i) {
        element[i] = std::tolower(static_cast<unsigned char>(element[i]));
    }
    return element;
}

int column_index(const std::vector<std::string>& headers, const std::string& name) {
    for (size_t i = 0; i < headers.size(); ++i) {
        if (lower(headers[i]) == name) return static_cast<int>(i);
    }
    return -1;
}

bool starts_control(const std::string& line) {
    const std::string value = lower(trim(line));
    return value == "loop_" || value.rfind("data_", 0) == 0 ||
           (!value.empty() && value[0] == '_');
}

} // namespace

StruData StruParser::parse(const std::string& filename) {
    StruData data;
    std::ifstream f(filename);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    while (std::getline(f, line)) {
        // Simple trimmer
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty() || line[0] == '#') continue;

        if (line == "ATOMIC_SPECIES") {
            while (std::getline(f, line)) {
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                if (line.empty() || line[0] == '#') continue;
                if (line == "NUMERICAL_ORBITAL" || line == "LATTICE_CONSTANT" || line == "LATTICE_VECTORS" || line == "ATOMIC_POSITIONS") break;

                std::stringstream ss(line);
                Species s;
                if (ss >> s.element >> s.mass >> s.pseudo_pot) {
                    data.species.push_back(s);
                }
                if (data.species.size() > 0 && (line.find("NUMERICAL_ORBITAL") != std::string::npos)) break;
            }
        }
        
        if (line == "NUMERICAL_ORBITAL") {
            int i = 0;
            while (std::getline(f, line)) {
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                if (line.empty() || line[0] == '#') continue;
                if (line == "LATTICE_CONSTANT" || line == "LATTICE_VECTORS" || line == "ATOMIC_POSITIONS") break;
                if (i < data.species.size()) {
                    data.species[i].orbital_file = line;
                    i++;
                }
            }
        }

        if (line == "LATTICE_CONSTANT") {
            while (std::getline(f, line)) {
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                if (line.empty() || line[0] == '#') continue;
                data.lattice_constant = std::stod(line.substr(0, line.find_first_of(" \t#")));
                // Convert Bohr to Angstrom (match ASE)
                data.lattice_constant *= 0.529177210903;
                break;
            }
        }

        if (line == "LATTICE_VECTORS") {
            for (int i = 0; i < 3; ++i) {
                while (std::getline(f, line)) {
                    line.erase(0, line.find_first_not_of(" \t\r\n"));
                    line.erase(line.find_last_not_of(" \t\r\n") + 1);
                    if (line.empty() || line[0] == '#') continue;
                    std::stringstream ss(line);
                    ss >> data.lattice_vectors(i, 0) >> data.lattice_vectors(i, 1) >> data.lattice_vectors(i, 2);
                    break;
                }
            }
        }

        if (line == "ATOMIC_POSITIONS") {
            std::string coord_type;
            while (std::getline(f, line)) {
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                if (line.empty() || line[0] == '#') continue;
                coord_type = line;
                break;
            }
            if (coord_type == "Direct") data.is_direct = true;
            
            for (auto& s : data.species) {
                // Find species name header
                while (std::getline(f, line)) {
                    line.erase(0, line.find_first_not_of(" \t\r\n"));
                    line.erase(line.find_last_not_of(" \t\r\n") + 1);
                    if (line.empty() || line[0] == '#') continue;
                    std::stringstream ss(line);
                    std::string species_name;
                    ss >> species_name;
                    if (species_name == s.element) break;
                }
                // Magnetic moment
                while (std::getline(f, line)) {
                    line.erase(0, line.find_first_not_of(" \t\r\n"));
                    line.erase(line.find_last_not_of(" \t\r\n") + 1);
                    if (line.empty() || line[0] == '#') continue;
                    break; 
                }
                // Number of atoms
                int num_atoms = 0;
                while (std::getline(f, line)) {
                    line.erase(0, line.find_first_not_of(" \t\r\n"));
                    line.erase(line.find_last_not_of(" \t\r\n") + 1);
                    if (line.empty() || line[0] == '#') continue;
                    num_atoms = std::stoi(line);
                    break;
                }
                for (int i = 0; i < num_atoms; ++i) {
                    while (std::getline(f, line)) {
                        line.erase(0, line.find_first_not_of(" \t\r\n"));
                        line.erase(line.find_last_not_of(" \t\r\n") + 1);
                        if (line.empty() || line[0] == '#') continue;
                        Atom a;
                        a.element = s.element;
                        std::stringstream ss(line);
                        ss >> a.pos(0) >> a.pos(1) >> a.pos(2);
                        data.atoms.push_back(a);
                        break;
                    }
                }
            }
        }
    }
    return data;
}

StruData StruParser::parse_cif(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) lines.push_back(line);

    const double unset = std::numeric_limits<double>::quiet_NaN();
    double a = unset, b = unset, c = unset;
    double alpha = unset, beta = unset, gamma = unset;
    for (const auto& raw : lines) {
        const auto tokens = cif_tokens(raw);
        if (tokens.size() < 2) continue;
        const auto key = lower(tokens[0]);
        if (key == "_cell_length_a") a = cif_number(tokens[1], key);
        else if (key == "_cell_length_b") b = cif_number(tokens[1], key);
        else if (key == "_cell_length_c") c = cif_number(tokens[1], key);
        else if (key == "_cell_angle_alpha") alpha = cif_number(tokens[1], key);
        else if (key == "_cell_angle_beta") beta = cif_number(tokens[1], key);
        else if (key == "_cell_angle_gamma") gamma = cif_number(tokens[1], key);
    }
    if (std::isnan(a) || std::isnan(b) || std::isnan(c) ||
        std::isnan(alpha) || std::isnan(beta) || std::isnan(gamma)) {
        throw std::runtime_error("CIF file does not define complete cell lengths and angles");
    }

    StruData data;
    data.lattice_constant = 1.0; // CIF cell lengths are Angstrom values.
    const double radians = std::acos(-1.0) / 180.0;
    const double ca = std::cos(alpha * radians);
    const double cb = std::cos(beta * radians);
    const double cg = std::cos(gamma * radians);
    const double sg = std::sin(gamma * radians);
    if (std::abs(sg) < 1e-12) {
        throw std::runtime_error("Invalid CIF cell: gamma angle makes lattice singular");
    }
    data.lattice_vectors.row(0) << a, 0.0, 0.0;
    data.lattice_vectors.row(1) << b * cg, b * sg, 0.0;
    const double cx = c * cb;
    const double cy = c * (ca - cb * cg) / sg;
    const double cz_sq = c * c - cx * cx - cy * cy;
    if (cz_sq < -1e-10) {
        throw std::runtime_error("Invalid CIF cell: lattice angles do not form a real cell");
    }
    data.lattice_vectors.row(2) << cx, cy, std::sqrt(std::max(0.0, cz_sq));

    std::unordered_set<std::string> species_seen;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (lower(trim(lines[i])) != "loop_") continue;

        std::vector<std::string> headers;
        size_t row_start = i + 1;
        while (row_start < lines.size()) {
            auto tokens = cif_tokens(lines[row_start]);
            if (tokens.empty()) {
                ++row_start;
                continue;
            }
            if (tokens[0].empty() || tokens[0][0] != '_') break;
            headers.push_back(tokens[0]);
            ++row_start;
        }

        int symbol_col = column_index(headers, "_atom_site_type_symbol");
        if (symbol_col < 0) symbol_col = column_index(headers, "_atom_site_label");
        const int fx = column_index(headers, "_atom_site_fract_x");
        const int fy = column_index(headers, "_atom_site_fract_y");
        const int fz = column_index(headers, "_atom_site_fract_z");
        const int cx_col = column_index(headers, "_atom_site_cartn_x");
        const int cy_col = column_index(headers, "_atom_site_cartn_y");
        const int cz_col = column_index(headers, "_atom_site_cartn_z");
        const bool fractional = fx >= 0 && fy >= 0 && fz >= 0;
        const bool cartesian = cx_col >= 0 && cy_col >= 0 && cz_col >= 0;
        if (symbol_col < 0 || (!fractional && !cartesian)) continue;

        data.is_direct = fractional;
        std::vector<std::string> values;
        size_t j = row_start;
        for (; j < lines.size() && !starts_control(lines[j]); ++j) {
            const auto row_tokens = cif_tokens(lines[j]);
            values.insert(values.end(), row_tokens.begin(), row_tokens.end());
            while (values.size() >= headers.size()) {
                Atom atom;
                atom.element = element_from_cif(values[symbol_col]);
                const int x_col = fractional ? fx : cx_col;
                const int y_col = fractional ? fy : cy_col;
                const int z_col = fractional ? fz : cz_col;
                atom.pos << cif_number(values[x_col], headers[x_col]),
                            cif_number(values[y_col], headers[y_col]),
                            cif_number(values[z_col], headers[z_col]);
                atom.move[0] = atom.move[1] = atom.move[2] = 1;
                data.atoms.push_back(atom);
                if (species_seen.insert(atom.element).second) {
                    data.species.push_back({atom.element, 0.0, "", ""});
                }
                values.erase(values.begin(), values.begin() + headers.size());
            }
        }
        if (!values.empty()) {
            throw std::runtime_error("Incomplete atom row in CIF file");
        }
        i = j == 0 ? j : j - 1;
    }
    if (data.atoms.empty()) {
        throw std::runtime_error("CIF file contains no readable atom-site coordinate loop");
    }
    return data;
}
