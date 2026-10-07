#include <etx/std.hxx>
#include <etx/core/core.hxx>
#include "scene_pbrt_parser.hxx"
#include "scene_pbrt_input.hxx"

#include <charconv>
#include <fstream>
#include <stdexcept>
#include <string_view>

namespace etx {
namespace {

enum class TokenKind { End, Word, String, OpenBracket, CloseBracket };

struct Token {
  TokenKind kind = TokenKind::End;
  std::string_view text;
  uint32_t line = 1u;
};

struct Lexer {
  explicit Lexer(const std::filesystem::path& file)
    : _file(file) {
    if (is_pbrt_gzip_file(file))
      _contents = read_pbrt_gzip(file);
    else {
      std::ifstream stream(file, std::ios::binary | std::ios::ate);
      if (stream.is_open() == false)
        fail(1u, "Cannot open PBRT input.");
      const auto size = stream.tellg();
      if (size < 0)
        fail(1u, "Cannot determine PBRT input size.");
      _contents.resize(static_cast<size_t>(size));
      stream.seekg(0);
      if (stream.read(_contents.data(), static_cast<std::streamsize>(_contents.size())).fail())
        fail(1u, "Cannot read PBRT input.");
    }
    if (_contents.starts_with("\xef\xbb\xbf"))
      _position = 3u;
    advance();
  }

  const Token& token() const {
    return _token;
  }

  [[noreturn]] void fail(uint32_t line, const std::string& message) const {
    throw std::runtime_error(PbrtLocation{_file, line}.describe() + ": " + message);
  }

  void advance() {
    skip_space();
    _token = {TokenKind::End, {}, _line};
    if (_position == _contents.size())
      return;
    const char first = _contents[_position++];
    if ((first == '[') || (first == ']')) {
      _token.kind = first == '[' ? TokenKind::OpenBracket : TokenKind::CloseBracket;
      return;
    }
    if (first == '"') {
      _token.kind = TokenKind::String;
      const size_t begin = _position;
      while (_position < _contents.size()) {
        const char character = _contents[_position++];
        if (character == '"') {
          _token.text = std::string_view(_contents).substr(begin, _position - begin - 1u);
          return;
        }
        if (character == '\\') {
          if (_position == _contents.size())
            break;
          if (_contents[_position++] == '\n')
            ++_line;
        } else if (character == '\n')
          ++_line;
      }
      fail(_token.line, "Unterminated string.");
    }
    _token.kind = TokenKind::Word;
    const size_t begin = _position - 1u;
    while (_position < _contents.size()) {
      const char character = _contents[_position];
      if (is_space(character) || (character == '#') || (character == '[') || (character == ']') || (character == '"'))
        break;
      ++_position;
    }
    _token.text = std::string_view(_contents).substr(begin, _position - begin);
  }

  std::string string() {
    if (_token.kind != TokenKind::String)
      fail(_token.line, "Expected a quoted string.");
    std::string result;
    result.reserve(_token.text.size());
    for (size_t index = 0u; index < _token.text.size(); ++index) {
      char character = _token.text[index];
      if (character == '\\') {
        character = _token.text[++index];
        switch (character) {
          case 'n':
            character = '\n';
            break;
          case 'r':
            character = '\r';
            break;
          case 't':
            character = '\t';
            break;
          case 'b':
            character = '\b';
            break;
          case 'f':
            character = '\f';
            break;
          case '\\':
          case '"':
            break;
          default:
            fail(_token.line, "Invalid string escape.");
        }
      }
      result.push_back(character);
    }
    advance();
    return result;
  }

  double number() {
    if (_token.kind != TokenKind::Word)
      fail(_token.line, "Expected a number.");
    auto text = _token.text;
    if (text.starts_with('+'))
      text.remove_prefix(1u);
    double result = 0.0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if ((parsed.ec != std::errc{}) || (parsed.ptr != (text.data() + text.size())) || (std::isfinite(result) == false))
      fail(_token.line, "Invalid number: " + std::string(_token.text));
    advance();
    return result;
  }

 private:
  static bool is_space(char character) {
    return (character == ' ') || (character == '\t') || (character == '\n') || (character == '\r') || (character == '\f');
  }

  void skip_space() {
    while (_position < _contents.size()) {
      const char character = _contents[_position];
      if (character == '#') {
        while ((_position < _contents.size()) && (_contents[_position] != '\n'))
          ++_position;
      } else if (is_space(character)) {
        ++_position;
        if (character == '\n')
          ++_line;
      } else
        return;
    }
  }

  std::filesystem::path _file;
  std::string _contents;
  size_t _position = 0u;
  uint32_t _line = 1u;
  Token _token;
};

struct Directive {
  std::string_view name;
  uint32_t strings;
  uint32_t numbers;
  bool parameters;
};

constexpr Directive kDirectives[] = {
  {"Accelerator", 1u, 0u, true},
  {"ActiveTransform", 0u, 0u, false},
  {"AreaLightSource", 1u, 0u, true},
  {"Attribute", 1u, 0u, true},
  {"AttributeBegin", 0u, 0u, false},
  {"AttributeEnd", 0u, 0u, false},
  {"Camera", 1u, 0u, true},
  {"ColorSpace", 1u, 0u, false},
  {"ConcatTransform", 0u, 16u, false},
  {"CoordinateSystem", 1u, 0u, false},
  {"CoordSysTransform", 1u, 0u, false},
  {"Film", 1u, 0u, true},
  {"Identity", 0u, 0u, false},
  {"Import", 1u, 0u, false},
  {"Include", 1u, 0u, false},
  {"Integrator", 1u, 0u, true},
  {"LightSource", 1u, 0u, true},
  {"LookAt", 0u, 9u, false},
  {"MakeNamedMaterial", 1u, 0u, true},
  {"MakeNamedMedium", 1u, 0u, true},
  {"Material", 1u, 0u, true},
  {"MediumInterface", 2u, 0u, false},
  {"NamedMaterial", 1u, 0u, false},
  {"ObjectBegin", 1u, 0u, false},
  {"ObjectEnd", 0u, 0u, false},
  {"ObjectInstance", 1u, 0u, false},
  {"Option", 0u, 0u, true},
  {"PixelFilter", 1u, 0u, true},
  {"ReverseOrientation", 0u, 0u, false},
  {"Rotate", 0u, 4u, false},
  {"Sampler", 1u, 0u, true},
  {"Scale", 0u, 3u, false},
  {"Shape", 1u, 0u, true},
  {"Texture", 3u, 0u, true},
  {"Transform", 0u, 16u, false},
  {"TransformBegin", 0u, 0u, false},
  {"TransformEnd", 0u, 0u, false},
  {"TransformTimes", 0u, 2u, false},
  {"Translate", 0u, 3u, false},
  {"WorldBegin", 0u, 0u, false},
  {"WorldEnd", 0u, 0u, false},
};

void read_parameter(Lexer& lexer, PbrtStatement& statement) {
  const uint32_t line = lexer.token().line;
  const std::string declaration = lexer.string();
  const size_t separator = declaration.find_first_of(" \t");
  const size_t name_begin = declaration.find_first_not_of(" \t", separator);
  if ((separator == std::string::npos) || (name_begin == std::string::npos))
    lexer.fail(line, "Expected a parameter type and name.");
  PbrtParameter parameter;
  parameter.type = declaration.substr(0u, separator);
  parameter.name = declaration.substr(name_begin);
  const bool array = lexer.token().kind == TokenKind::OpenBracket;
  if (array)
    lexer.advance();
  const auto read_value = [&] {
    if (lexer.token().kind == TokenKind::String)
      parameter.strings.push_back(lexer.string());
    else if ((parameter.type == "bool") && ((lexer.token().text == "true") || (lexer.token().text == "false"))) {
      parameter.strings.emplace_back(lexer.token().text);
      lexer.advance();
    } else
      parameter.numbers.push_back(lexer.number());
  };
  if (array) {
    while (lexer.token().kind != TokenKind::CloseBracket) {
      if (lexer.token().kind == TokenKind::End)
        lexer.fail(line, "Unterminated parameter array.");
      read_value();
    }
    lexer.advance();
  } else
    read_value();
  if ((parameter.numbers.empty() == false) && (parameter.strings.empty() == false))
    lexer.fail(line, "A parameter array cannot mix strings and numbers.");
  statement.parameters.push_back(std::move(parameter));
}

void visit_source(const std::filesystem::path& file, const std::filesystem::path& root, bool expand_includes, const PbrtVisitor& visitor,
  std::vector<std::filesystem::path>& stack) {
  const auto canonical = std::filesystem::weakly_canonical(file);
  if (std::find(stack.begin(), stack.end(), canonical) != stack.end())
    throw std::runtime_error(path_to_utf8(file) + ": Cyclic PBRT include.");
  stack.push_back(canonical);
  Lexer lexer(file);
  while (lexer.token().kind != TokenKind::End) {
    PbrtStatement statement;
    statement.location = {file, lexer.token().line};
    statement.directive = lexer.token().text;
    const auto definition = std::find_if(std::begin(kDirectives), std::end(kDirectives), [&](const Directive& directive) {
      return directive.name == statement.directive;
    });
    if ((lexer.token().kind != TokenKind::Word) || (definition == std::end(kDirectives)))
      lexer.fail(lexer.token().line, "Unknown PBRT directive: " + statement.directive);
    lexer.advance();
    for (uint32_t index = 0u; index < definition->strings; ++index)
      statement.arguments.push_back(lexer.string());
    if (statement.directive == "ActiveTransform") {
      const auto& token = lexer.token();
      if ((token.kind != TokenKind::Word) || ((token.text != "All") && (token.text != "StartTime") && (token.text != "EndTime")))
        lexer.fail(token.line, "Expected All, StartTime, or EndTime.");
      statement.arguments.emplace_back(token.text);
      lexer.advance();
    }
    if (definition->numbers > 0u) {
      const bool array = lexer.token().kind == TokenKind::OpenBracket;
      if (array)
        lexer.advance();
      statement.numbers.reserve(definition->numbers);
      for (uint32_t index = 0u; index < definition->numbers; ++index)
        statement.numbers.push_back(lexer.number());
      if (array) {
        if (lexer.token().kind != TokenKind::CloseBracket)
          lexer.fail(lexer.token().line, "Unexpected transform array size.");
        lexer.advance();
      }
    }
    if (definition->parameters) {
      while (lexer.token().kind == TokenKind::String)
        read_parameter(lexer, statement);
    }
    visitor(statement);
    if (expand_includes && (statement.directive == "Include"))
      visit_source(root / std::filesystem::u8path(statement.arguments[0]), root, true, visitor, stack);
  }
  stack.pop_back();
}

}  // namespace

std::string PbrtLocation::describe() const {
  return path_to_utf8(file) + ":" + std::to_string(line);
}

const PbrtParameter* PbrtStatement::find(const char* name) const {
  for (auto iterator = parameters.rbegin(); iterator != parameters.rend(); ++iterator) {
    if (iterator->name == name)
      return &*iterator;
  }
  return nullptr;
}

void visit_pbrt_file(const std::filesystem::path& file, bool expand_includes, const PbrtVisitor& visitor) {
  const auto absolute = std::filesystem::absolute(file).lexically_normal();
  std::vector<std::filesystem::path> stack;
  visit_source(absolute, absolute.parent_path(), expand_includes, visitor, stack);
}

}  // namespace etx
