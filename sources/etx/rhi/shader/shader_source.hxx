#pragma once

#include <etx/core/core.hxx>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace etx {

// HLSL has no function pointers. Retain every overload of a referenced name and
// referenced plain struct, preserving resource bindings and other declarations.
// Unsupported type syntax and nested scopes stay intact as conservative roots.
struct ShaderSourceFingerprint {
  explicit ShaderSourceFingerprint(std::string_view source)
    : _source(source) {
    tokenize(source);
    find_declarations();
  }

  uint64_t entry_hash(std::string_view entry_point) const {
    const std::vector<bool> retained = retained_declarations(entry_point);
    uint64_t hash = 0u;
    size_t declaration_index = 0u;
    for (size_t index = 0u; index < _tokens.size();) {
      if ((declaration_index < _declarations.size()) && (index == _declarations[declaration_index].begin)) {
        const size_t current_declaration = declaration_index++;
        if (retained[current_declaration] == false) {
          index = _declarations[current_declaration].end;
          continue;
        }
      }
      const std::string_view token = _tokens[index++];
      const uint64_t size = token.size();
      hash = etx_hash64_continue(&size, sizeof(size), hash);
      hash = etx_hash64_continue(token.data(), token.size(), hash);
    }
    return hash;
  }

  std::string entry_source(std::string_view entry_point) const {
    const std::vector<bool> retained = retained_declarations(entry_point);
    std::string result;
    result.reserve(_source.size());
    size_t begin = 0u;
    for (size_t index = 0u; index < _declarations.size(); ++index) {
      if (retained[index]) {
        continue;
      }
      const Declaration& declaration = _declarations[index];
      const size_t declaration_begin = static_cast<size_t>(_tokens[declaration.begin].data() - _source.data());
      const size_t declaration_end = static_cast<size_t>(_tokens[declaration.end - 1u].data() - _source.data()) + _tokens[declaration.end - 1u].size();
      result.append(_source.substr(begin, declaration_begin - begin));
      result += '\n';
      begin = declaration_end;
    }
    result.append(_source.substr(begin));
    return result;
  }

 private:
  struct Declaration {
    size_t begin;
    size_t end;
  };
  std::vector<std::string_view> _tokens;
  std::vector<Declaration> _declarations;
  std::vector<size_t> _prototype_name_tokens;
  std::unordered_map<std::string_view, std::vector<size_t>> _declaration_names;

  std::string_view _source;

  std::vector<bool> retained_declarations(std::string_view entry_point) const {
    std::vector<bool> retained(_declarations.size(), false);
    std::vector<size_t> pending;
    auto retain_name = [&](std::string_view name) {
      const auto it = _declaration_names.find(name);
      if (it != _declaration_names.end()) {
        for (const size_t index : it->second) {
          if (retained[index] == false) {
            retained[index] = true;
            pending.push_back(index);
          }
        }
      }
    };
    retain_name(entry_point);
    size_t declaration_index = 0u;
    size_t prototype_index = 0u;
    for (size_t index = 0u; index < _tokens.size();) {
      if ((declaration_index < _declarations.size()) && (index == _declarations[declaration_index].begin)) {
        index = _declarations[declaration_index++].end;
      } else if ((prototype_index < _prototype_name_tokens.size()) && (index == _prototype_name_tokens[prototype_index])) {
        // A declaration does not call its function. Keep other references,
        // including helpers used by default arguments, as conservative roots.
        ++prototype_index;
        ++index;
      } else {
        retain_name(_tokens[index++]);
      }
    }
    for (size_t next = 0u; next < pending.size(); ++next) {
      const Declaration& declaration = _declarations[pending[next]];
      for (size_t index = declaration.begin; index < declaration.end; ++index) {
        retain_name(_tokens[index]);
      }
    }

    return retained;
  }

  static bool identifier_start(char ch) {
    return ((ch >= 'a') && (ch <= 'z')) || ((ch >= 'A') && (ch <= 'Z')) || (ch == '_') || (static_cast<unsigned char>(ch) >= 128u);
  }

  static bool digit(char ch) {
    return (ch >= '0') && (ch <= '9');
  }

  void tokenize(std::string_view source) {
    _tokens.reserve(source.size() / 4u);
    bool line_start = true;
    for (size_t position = 0u; position < source.size();) {
      const char ch = source[position];
      if ((ch == ' ') || (ch == '\t') || (ch == '\r') || (ch == '\n')) {
        line_start = line_start || (ch == '\n');
        ++position;
        continue;
      }
      if ((ch == '#') && line_start) {
        size_t directive = position + 1u;
        while ((directive < source.size()) && ((source[directive] == ' ') || (source[directive] == '\t'))) {
          ++directive;
        }
        // Preprocessing already expands __LINE__/__FILE__; location directives
        // affect diagnostics only when shader debug information is disabled.
        if ((source.substr(directive, 4u) == "line") || ((directive < source.size()) && digit(source[directive]))) {
          position = source.find('\n', directive);
          if (position == std::string_view::npos) {
            break;
          }
          continue;
        }
      }
      if ((ch == '/') && ((position + 1u) < source.size())) {
        if (source[position + 1u] == '/') {
          position = source.find('\n', position + 2u);
          if (position == std::string_view::npos) {
            break;
          }
          continue;
        }
        if (source[position + 1u] == '*') {
          const size_t end = source.find("*/", position + 2u);
          if (end != std::string_view::npos) {
            line_start = line_start || (source.substr(position, end - position).find('\n') != std::string_view::npos);
            position = end + 2u;
            continue;
          }
        }
      }
      line_start = false;
      const size_t begin = position++;
      if (identifier_start(ch)) {
        while ((position < source.size()) && (identifier_start(source[position]) || digit(source[position]))) {
          ++position;
        }
      } else if (digit(ch) || ((ch == '.') && (position < source.size()) && digit(source[position]))) {
        while (position < source.size()) {
          const char next = source[position];
          const char previous = source[position - 1u];
          if (identifier_start(next) || digit(next) || (next == '.') ||
              (((next == '+') || (next == '-')) && ((previous == 'e') || (previous == 'E') || (previous == 'p') || (previous == 'P')))) {
            ++position;
          } else {
            break;
          }
        }
      } else if ((ch == '"') || (ch == '\'')) {
        while (position < source.size()) {
          const char next = source[position++];
          if ((next == '\\') && (position < source.size())) {
            ++position;
          } else if (next == ch) {
            break;
          }
        }
      } else {
        constexpr std::string_view operators[] = {"<<=", ">>=", "...", "++", "--", "->", "::", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
          "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "##"};
        for (const std::string_view op : operators) {
          if (source.substr(begin, op.size()) == op) {
            position = begin + op.size();
            break;
          }
        }
      }
      _tokens.push_back(source.substr(begin, position - begin));
    }
  }

  void find_declarations() {
    std::unordered_map<std::string_view, size_t> struct_name_counts;
    for (size_t index = 0u; (index + 1u) < _tokens.size(); ++index) {
      if ((_tokens[index] == "struct") && identifier_start(_tokens[index + 1u][0])) {
        ++struct_name_counts[_tokens[index + 1u]];
      }
    }
    size_t declaration_begin = 0u;
    for (size_t index = 0u; index < _tokens.size(); ++index) {
      if (_tokens[index] == ";") {
        const size_t name_token = function_name_token(declaration_begin, index);
        if (name_token != std::string_view::npos) {
          _prototype_name_tokens.push_back(name_token);
        }
        declaration_begin = index + 1u;
        continue;
      }
      if (_tokens[index] != "{") {
        continue;
      }
      const size_t body_begin = index;
      size_t depth = 1u;
      while ((depth > 0u) && ((index + 1u) < _tokens.size())) {
        const std::string_view token = _tokens[++index];
        if (token == "{") {
          ++depth;
        } else if (token == "}") {
          --depth;
        }
      }
      if (depth != 0u) {
        _declarations.clear();
        _declaration_names.clear();
        _prototype_name_tokens.clear();
        return;
      }
      const size_t name_token = function_name_token(declaration_begin, body_begin);
      if (name_token != std::string_view::npos) {
        _declaration_names[_tokens[name_token]].push_back(_declarations.size());
        _declarations.push_back({declaration_begin, index + 1u});
      } else if (plain_struct(declaration_begin, body_begin, index) && (struct_name_counts[_tokens[declaration_begin + 1u]] == 1u)) {
        _declaration_names[_tokens[declaration_begin + 1u]].push_back(_declarations.size());
        _declarations.push_back({declaration_begin, index + 2u});
        ++index;
      }
      declaration_begin = index + 1u;
    }
  }

  bool plain_struct(size_t declaration_begin, size_t body_begin, size_t body_end) const {
    if ((body_begin != (declaration_begin + 2u)) || (_tokens[declaration_begin] != "struct") || (identifier_start(_tokens[declaration_begin + 1u][0]) == false) ||
        ((body_end + 1u) >= _tokens.size()) || (_tokens[body_end + 1u] != ";") || ((body_end > (body_begin + 1u)) && (_tokens[body_end - 1u] != ";"))) {
      return false;
    }
    bool initializer = false;
    uint32_t brackets = 0u;
    uint32_t parentheses = 0u;
    for (size_t index = body_begin + 1u; index < body_end; ++index) {
      const std::string_view token = _tokens[index];
      if ((token == "struct") || (token == "class") || (token == "enum") || (token == "typedef") || (token == "using") || (token == "template") || (token == "namespace") ||
          (token == "static") || (token == "cbuffer") || (token == "tbuffer") || (token == "interface") || (token == "union") || (token == "::") || (token == "#") ||
          (token == "{") || (token == "}") || (token == "<") || (token == ">")) {
        return false;
      }
      if (token == "[") {
        ++brackets;
      } else if (token == "]") {
        if (brackets == 0u) {
          return false;
        }
        --brackets;
      } else if (token == "(") {
        if ((initializer == false) && (brackets == 0u)) {
          return false;
        }
        ++parentheses;
      } else if (token == ")") {
        if (parentheses == 0u) {
          return false;
        }
        --parentheses;
      } else if ((brackets == 0u) && (parentheses == 0u)) {
        if (token == "=") {
          initializer = true;
        } else if ((token == ";") || (token == ",")) {
          initializer = false;
        }
      }
    }
    return (brackets == 0u) && (parentheses == 0u);
  }

  size_t function_name_token(size_t declaration_begin, size_t declaration_end) const {
    size_t name_token = std::string_view::npos;
    uint32_t parentheses = 0u;
    uint32_t brackets = 0u;
    bool function_header = true;
    for (size_t header = declaration_begin; header < declaration_end; ++header) {
      const std::string_view token = _tokens[header];
      if (token == "[") {
        ++brackets;
      } else if ((token == "]") && (brackets > 0u)) {
        --brackets;
      } else if (brackets == 0u) {
        if ((parentheses == 0u) && ((token == "=") || (token == "struct") || (token == "class") || (token == "enum") || (token == "namespace") || (token == "cbuffer") ||
                                     (token == "tbuffer") || (token == "#"))) {
          function_header = false;
        }
        if (token == "(") {
          if ((parentheses == 0u) && (header > declaration_begin) && identifier_start(_tokens[header - 1u][0])) {
            name_token = header - 1u;
          }
          ++parentheses;
        } else if ((token == ")") && (parentheses > 0u)) {
          --parentheses;
        }
      }
    }
    if (function_header && (parentheses == 0u) && (brackets == 0u)) {
      return name_token;
    }
    return std::string_view::npos;
  }
};

}  // namespace etx
