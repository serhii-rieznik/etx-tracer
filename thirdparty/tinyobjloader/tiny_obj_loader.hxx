/*
The MIT License (MIT)

Copyright (c) 2012-Present, Syoyo Fujita and many contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

//
// version 2.0.0 : Add new object oriented API. 1.x API is still provided.
//                 * Support line primitive.
//                 * Support points primitive.
//                 * Support multiple search path for .mtl(v1 API).
//                 * Support escaped whitespece in mtllib
//                 * Add robust triangulation using Mapbox earcut(TINYOBJLOADER_USE_MAPBOX_EARCUT).
// version 1.4.0 : Modifed ParseTextureNameAndOption API
// version 1.3.1 : Make ParseTextureNameAndOption API public
// version 1.3.0 : Separate warning and error message(breaking API of LoadObj)
// version 1.2.3 : Added color space extension('-colorspace') to tex opts.
// version 1.2.2 : Parse multiple group names.
// version 1.2.1 : Added initial support for line('l') primitive(PR #178)
// version 1.2.0 : Hardened implementation(#175)
// version 1.1.1 : Support smoothing groups(#162)
// version 1.1.0 : Support parsing vertex color(#144)
// version 1.0.8 : Fix parsing `g` tag just after `usemtl`(#138)
// version 1.0.7 : Support multiple tex options(#126)
// version 1.0.6 : Add TINYOBJLOADER_USE_DOUBLE option(#124)
// version 1.0.5 : Ignore `Tr` when `d` exists in MTL(#43)
// version 1.0.4 : Support multiple filenames for 'mtllib'(#112)
// version 1.0.3 : Support parsing texture options(#85)
// version 1.0.2 : Improve parsing speed by about a factor of 2 for large
// files(#105)
// version 1.0.1 : Fixes a shape is lost if obj ends with a 'usemtl'(#104)
// version 1.0.0 : Change data structure. Change license from BSD to MIT.
//

#ifndef TINY_OBJ_LOADER_H_
#define TINY_OBJ_LOADER_H_

#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <charconv>

struct BufferedReader {
  struct Line {
    char* begin = nullptr;
    char* end = nullptr;

    bool empty() const {
      return end <= begin;
    }

    void trim() {
      while ((begin < end) && *begin && ((*begin == ' ') || (*begin == '\t')))
        begin++;
      while ((end > begin) && *end && ((*end == ' ') || (*end == '\t')))
        end--;
      *end = 0;
    }
  };

  BufferedReader(std::istream* stream, size_t buffer_size = 8 * 1024 * 1024)
    : stream_(stream)
    , buffer_size_(buffer_size)
    , buffer_pos_(0)
    , buffer_end_(0)
    , eof_(false) {
    buffer_.resize(buffer_size);
    refill_buffer();
  }

  bool getline(Line& line) {
    while ((eof_ == false) || (buffer_pos_ < buffer_end_)) {
      char* start = buffer_.data() + buffer_pos_;
      char* end = buffer_.data() + buffer_end_;
      char* line_end = start;
      while ((line_end < end) && (*line_end != '\n') && (*line_end != '\r')) {
        ++line_end;
      }

      if (line_end < end) {
        line = {start, line_end};
        if ((line_end < end) && (*line_end == '\r'))
          ++line_end;
        if ((line_end < end) && (*line_end == '\n'))
          ++line_end;
        line.trim();
        buffer_pos_ = line_end - buffer_.data();
        return true;
      }

      if (refill_buffer())
        continue;

      if (buffer_pos_ >= buffer_end_)
        return false;

      line = {start, end};
      line.trim();
      buffer_pos_ = buffer_end_;
      return true;
    }

    return false;
  }

 private:
  bool refill_buffer() {
    if (eof_)
      return false;

    // Move remaining data to beginning if needed
    size_t remaining = buffer_end_ - buffer_pos_;
    if (remaining > 0 && buffer_pos_ > 0) {
      std::memmove(buffer_.data(), buffer_.data() + buffer_pos_, remaining);
    }

    buffer_pos_ = 0;
    buffer_end_ = remaining;

    // Read more data, but leave space for rollover
    size_t read_size = buffer_size_ - remaining;
    if (read_size > buffer_size_ / 4) {  // Keep some space for rollover
      read_size = buffer_size_ - buffer_size_ / 4;
    }

    stream_->read(buffer_.data() + remaining, read_size);
    size_t bytes_read = stream_->gcount();

    if (bytes_read == 0) {
      eof_ = true;
      return remaining > 0;
    }

    buffer_end_ = remaining + bytes_read;
    return true;
  }

  std::istream* stream_;
  size_t buffer_size_;
  std::vector<char> buffer_;
  size_t buffer_pos_;
  size_t buffer_end_;
  bool eof_;
};

#if !defined(_MSC_VER)
# define _strnicmp strncasecmp
#endif

namespace tinyobj {

// TODO(syoyo): Better C++11 detection for older compiler
#if __cplusplus > 199711L
# define TINYOBJ_OVERRIDE override
#else
# define TINYOBJ_OVERRIDE
#endif

#ifdef __clang__
# pragma clang diagnostic push
# if __has_warning("-Wzero-as-null-pointer-constant")
#  pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
# endif

# pragma clang diagnostic ignored "-Wpadded"

#endif

// https://en.wikipedia.org/wiki/Wavefront_.obj_file says ...
//
//  -blendu on | off                       # set horizontal texture blending
//  (default on)
//  -blendv on | off                       # set vertical texture blending
//  (default on)
//  -boost real_value                      # boost mip-map sharpness
//  -mm base_value gain_value              # modify texture map values (default
//  0 1)
//                                         #     base_value = brightness,
//                                         gain_value = contrast
//  -o u [v [w]]                           # Origin offset             (default
//  0 0 0)
//  -s u [v [w]]                           # Scale                     (default
//  1 1 1)
//  -t u [v [w]]                           # Turbulence                (default
//  0 0 0)
//  -texres resolution                     # texture resolution to create
//  -clamp on | off                        # only render texels in the clamped
//  0-1 range (default off)
//                                         #   When unclamped, textures are
//                                         repeated across a surface,
//                                         #   when clamped, only texels which
//                                         fall within the 0-1
//                                         #   range are rendered.
//  -bm mult_value                         # bump multiplier (for bump maps
//  only)
//
//  -imfchan r | g | b | m | l | z         # specifies which channel of the file
//  is used to
//                                         # create a scalar or bump texture.
//                                         r:red, g:green,
//                                         # b:blue, m:matte, l:luminance,
//                                         z:z-depth..
//                                         # (the default for bump is 'l' and
//                                         for decal is 'm')
//  bump -imfchan r bumpmap.tga            # says to use the red channel of
//  bumpmap.tga as the bumpmap
//
// For reflection maps...
//
//   -type sphere                           # specifies a sphere for a "refl"
//   reflection map
//   -type cube_top    | cube_bottom |      # when using a cube map, the texture
//   file for each
//         cube_front  | cube_back   |      # side of the cube is specified
//         separately
//         cube_left   | cube_right
//
// TinyObjLoader extension.
//
//   -colorspace SPACE                      # Color space of the texture. e.g.
//   'sRGB` or 'linear'
//

#ifdef TINYOBJLOADER_USE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

struct material_t {
  std::string name;
  std::vector<std::pair<std::string, std::string>> unknown_parameter;
};

struct index_t {
  int vertex_index;
  int normal_index;
  int texcoord_index;
};

struct mesh_t {
  std::vector<index_t> indices;
  std::vector<unsigned char> num_face_vertices;
  std::vector<unsigned int> smoothing_group_ids;
};

struct vertex_index_t {
  int v_idx, vt_idx, vn_idx;
  vertex_index_t()
    : v_idx(-1)
    , vt_idx(-1)
    , vn_idx(-1) {
  }
  explicit vertex_index_t(int idx)
    : v_idx(idx)
    , vt_idx(idx)
    , vn_idx(idx) {
  }
  vertex_index_t(int vidx, int vtidx, int vnidx)
    : v_idx(vidx)
    , vt_idx(vtidx)
    , vn_idx(vnidx) {
  }
};

struct face_t {
  unsigned int smoothing_group_id;             // smoothing group id. 0 = smoothing groupd is off.
  std::vector<vertex_index_t> vertex_indices;  // face vertex indices.

  face_t()
    : smoothing_group_id(0) {
    vertex_indices.reserve(4);
  }
};

struct shape_t {
  std::string name;
  mesh_t mesh;
  std::vector<int> face_material_ids;  // material_id per face (same order as mesh faces)
};

// Vertex attributes
struct attrib_t {
  std::vector<real_t> vertex_x, vertex_y, vertex_z;
  std::vector<real_t> normal_x, normal_y, normal_z;
  std::vector<real_t> texcoord_u, texcoord_v;
  std::vector<real_t> texcoord_ws;
  std::vector<real_t> colors;

  attrib_t() {
  }
};

struct MaterialReader {
  MaterialReader() = default;
  virtual ~MaterialReader() = default;
  virtual bool operator()(const std::string& matId, std::vector<material_t>* materials, std::map<std::string, int>* matMap, std::string* warn, std::string* err) = 0;
};

///
/// Read .mtl from a file.
///
class MaterialFileReader : public MaterialReader {
 public:
  // Path could contain separator(';' in Windows, ':' in Posix)
  explicit MaterialFileReader(const std::string& mtl_basedir)
    : m_mtlBaseDir(mtl_basedir) {
  }
  virtual ~MaterialFileReader() TINYOBJ_OVERRIDE {
  }
  virtual bool operator()(const std::string& matId, std::vector<material_t>* materials, std::map<std::string, int>* matMap, std::string* warn, std::string* err) TINYOBJ_OVERRIDE;

 private:
  std::string m_mtlBaseDir;
};

bool LoadObj(attrib_t* attrib, std::vector<shape_t>* shapes, std::vector<material_t>* materials, std::string* warn, std::string* err, const char* filename,
  const char* mtl_basedir = NULL, const char* mtl_custom_file = NULL, bool triangulate = true, bool default_vcols_fallback = true);

bool LoadObj(attrib_t* attrib, std::vector<shape_t>* shapes, std::vector<material_t>* materials, std::string* warn, std::string* err, std::istream* inStream,
  MaterialReader* readMatFn, const char* customMaterials, bool triangulate = true, bool default_vcols_fallback = true);

void LoadMtl(std::map<std::string, int>* material_map, std::vector<material_t>* materials, std::istream* inStream, std::string* warning, std::string* err);

}  // namespace tinyobj

#endif  // TINY_OBJ_LOADER_H_

#define TINYOBJLOADER_IMPLEMENTATION 1

#ifdef TINYOBJLOADER_IMPLEMENTATION
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

namespace tinyobj {

// Convert vertex_index_t to index_t
static index_t to_index(const vertex_index_t& vi) {
  index_t idx;
  idx.vertex_index = vi.v_idx;
  idx.normal_index = vi.vn_idx;
  idx.texcoord_index = vi.vt_idx;
  return idx;
}

struct obj_shape {
  std::vector<real_t> v;
  std::vector<real_t> vn;
  std::vector<real_t> vt;
};

//
// Manages group of primitives(face, line, points, ...)
struct PrimGroup {
  std::vector<face_t> faceGroup;

  void clear() {
    faceGroup.clear();
  }

  bool IsEmpty() const {
    return faceGroup.empty();
  }

  // TODO(syoyo): bspline, surface, ...
};

#define IS_SPACE(x)    (((x) == ' ') || ((x) == '\t'))
#define IS_DIGIT(x)    (static_cast<unsigned int>((x) - '0') < static_cast<unsigned int>(10))
#define IS_NEW_LINE(x) (((x) == '\r') || ((x) == '\n') || ((x) == '\0'))

// Make index zero-base, and also support relative index.
static inline bool fixIndex(int idx, int n, int* ret) {
  if (!ret) {
    return false;
  }

  if (idx > 0) {
    (*ret) = idx - 1;
    return true;
  }

  if (idx == 0) {
    // zero is not allowed according to the spec.
    return false;
  }

  if (idx < 0) {
    (*ret) = n + idx;    // negative value = relative
    return (*ret) >= 0;  // bounds check for negative indices
  }

  return false;  // never reach here.
}

static inline std::string parseString(char** token) {
  std::string s;
  (*token) += strspn((*token), " \t");
  size_t e = strcspn((*token), " \t\r");
  s = std::string((*token), &(*token)[e]);
  (*token) += e;
  return s;
}

static inline int parseInt(char** token) {
  (*token) += strspn((*token), " \t");
  int i = atoi((*token));
  (*token) += strcspn((*token), " \t\r");
  return i;
}

static bool tryParseDouble(char* s, char* s_end, double* result) {
#if defined(__APPLE__) && defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 160000
  // On macOS with older deployment targets, std::from_chars for floating-point is not available
  char* endptr = nullptr;
  *result = strtod(s, &endptr);
  return endptr != nullptr && endptr <= s_end;
#else
  return uint32_t(std::from_chars(s, s_end, *result).ec) == 0;
#endif
}

static inline real_t parseReal(char** token, double default_value = 0.0) {
  (*token) += strspn((*token), " \t");
  char* end = (*token) + strcspn((*token), " \t\r");
  double val = default_value;
  tryParseDouble((*token), end, &val);
  real_t f = static_cast<real_t>(val);
  (*token) = end;
  return f;
}

static inline bool parseReal(char** token, real_t* out) {
  (*token) += strspn((*token), " \t");
  char* end = (*token) + strcspn((*token), " \t\r");
  double val;
  bool ret = tryParseDouble((*token), end, &val);
  if (ret) {
    real_t f = static_cast<real_t>(val);
    (*out) = f;
  }
  (*token) = end;
  return ret;
}

static inline void parseReal2(real_t* x, real_t* y, char** token, const double default_x = 0.0, const double default_y = 0.0) {
  (*x) = parseReal(token, default_x);
  (*y) = parseReal(token, default_y);
}

static inline void parseReal3(real_t* x, real_t* y, real_t* z, char** token, const double default_x = 0.0, const double default_y = 0.0, const double default_z = 0.0) {
  (*x) = parseReal(token, default_x);
  (*y) = parseReal(token, default_y);
  (*z) = parseReal(token, default_z);
}

static inline void parseV(real_t* x, real_t* y, real_t* z, real_t* w, char** token, const double default_x = 0.0, const double default_y = 0.0, const double default_z = 0.0,
  const double default_w = 1.0) {
  (*x) = parseReal(token, default_x);
  (*y) = parseReal(token, default_y);
  (*z) = parseReal(token, default_z);
  (*w) = parseReal(token, default_w);
}

// Extension: parse vertex with colors(6 items)
static inline bool parseVertexWithColor(real_t* x, real_t* y, real_t* z, real_t* r, real_t* g, real_t* b, char** token, const double default_x = 0.0, const double default_y = 0.0,
  const double default_z = 0.0) {
  (*x) = parseReal(token, default_x);
  (*y) = parseReal(token, default_y);
  (*z) = parseReal(token, default_z);

  const bool found_color = parseReal(token, r) && parseReal(token, g) && parseReal(token, b);

  if (!found_color) {
    (*r) = (*g) = (*b) = 1.0;
  }

  return found_color;
}

static inline bool parseOnOff(const char** token, bool default_value = true) {
  (*token) += strspn((*token), " \t");
  const char* end = (*token) + strcspn((*token), " \t\r");

  bool ret = default_value;
  if ((0 == _strnicmp((*token), "on", 2))) {
    ret = true;
  } else if ((0 == _strnicmp((*token), "off", 3))) {
    ret = false;
  }

  (*token) = end;
  return ret;
}

// Parse triples with index offsets: i, i/j/k, i//k, i/j
static bool parseTriple(char** token, int vsize, int vnsize, int vtsize, vertex_index_t* ret) {
  if (!ret) {
    return false;
  }

  vertex_index_t vi(-1);

  if (!fixIndex(atoi((*token)), vsize, &(vi.v_idx))) {
    return false;
  }

  (*token) += strcspn((*token), "/ \t\r");
  if ((*token)[0] != '/') {
    (*ret) = vi;
    return true;
  }
  (*token)++;

  // i//k
  if ((*token)[0] == '/') {
    (*token)++;
    if (!fixIndex(atoi((*token)), vnsize, &(vi.vn_idx))) {
      return false;
    }
    (*token) += strcspn((*token), "/ \t\r");
    (*ret) = vi;
    return true;
  }

  // i/j/k or i/j
  if (!fixIndex(atoi((*token)), vtsize, &(vi.vt_idx))) {
    return false;
  }

  (*token) += strcspn((*token), "/ \t\r");
  if ((*token)[0] != '/') {
    (*ret) = vi;
    return true;
  }

  // i/j/k
  (*token)++;  // skip '/'
  if (!fixIndex(atoi((*token)), vnsize, &(vi.vn_idx))) {
    return false;
  }
  (*token) += strcspn((*token), "/ \t\r");

  (*ret) = vi;

  return true;
}

// Parse raw triples: i, i/j/k, i//k, i/j
static vertex_index_t parseRawTriple(const char** token) {
  vertex_index_t vi(static_cast<int>(0));  // 0 is an invalid index in OBJ

  vi.v_idx = atoi((*token));
  (*token) += strcspn((*token), "/ \t\r");
  if ((*token)[0] != '/') {
    return vi;
  }
  (*token)++;

  // i//k
  if ((*token)[0] == '/') {
    (*token)++;
    vi.vn_idx = atoi((*token));
    (*token) += strcspn((*token), "/ \t\r");
    return vi;
  }

  // i/j/k or i/j
  vi.vt_idx = atoi((*token));
  (*token) += strcspn((*token), "/ \t\r");
  if ((*token)[0] != '/') {
    return vi;
  }

  // i/j/k
  (*token)++;  // skip '/'
  vi.vn_idx = atoi((*token));
  (*token) += strcspn((*token), "/ \t\r");
  return vi;
}

static void InitMaterial(material_t* material) {
  material->name = "";
  material->unknown_parameter.clear();
}

// code from https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html
template <typename T>
static int pnpoly(int nvert, T* vertx, T* verty, T testx, T testy) {
  int i, j, c = 0;
  for (i = 0, j = nvert - 1; i < nvert; j = i++) {
    if (((verty[i] > testy) != (verty[j] > testy)) && (testx < (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) + vertx[i]))
      c = !c;
  }
  return c;
}

// TODO(syoyo): refactor function.
static bool exportGroupsToShape(shape_t* shape, const PrimGroup& prim_group, const std::string& name, bool triangulate, const std::vector<real_t>& vertex_x,
  const std::vector<real_t>& vertex_y, const std::vector<real_t>& vertex_z, int current_material_id, std::string* warn) {
  if (prim_group.IsEmpty()) {
    return false;
  }

  shape->name = name;

  // Track material for each face
  size_t current_face_index = 0;

  // polygon
  if (!prim_group.faceGroup.empty()) {
    // Flatten vertices and indices
    for (size_t i = 0; i < prim_group.faceGroup.size(); i++) {
      const face_t& face = prim_group.faceGroup[i];

      size_t npolys = face.vertex_indices.size();

      if (npolys < 3) {
        // Face must have 3+ vertices.
        if (warn) {
          (*warn) += "Degenerated face found\n.";
        }
        continue;
      }

      // Store material_id for this face
      shape->face_material_ids.push_back(current_material_id);

      if (triangulate) {
        if (npolys == 4) {
          vertex_index_t i0 = face.vertex_indices[0];
          vertex_index_t i1 = face.vertex_indices[1];
          vertex_index_t i2 = face.vertex_indices[2];
          vertex_index_t i3 = face.vertex_indices[3];

          size_t vi0 = size_t(i0.v_idx);
          size_t vi1 = size_t(i1.v_idx);
          size_t vi2 = size_t(i2.v_idx);
          size_t vi3 = size_t(i3.v_idx);

          if ((vi0 >= vertex_x.size()) || (vi1 >= vertex_x.size()) || (vi2 >= vertex_x.size()) || (vi3 >= vertex_x.size())) {
            // Invalid triangle.
            // FIXME(syoyo): Is it ok to simply skip this invalid triangle?
            if (warn) {
              (*warn) += "Face with invalid vertex index found.\n";
            }
            continue;
          }

          real_t v0x = vertex_x[vi0];
          real_t v0y = vertex_y[vi0];
          real_t v0z = vertex_z[vi0];
          real_t v1x = vertex_x[vi1];
          real_t v1y = vertex_y[vi1];
          real_t v1z = vertex_z[vi1];
          real_t v2x = vertex_x[vi2];
          real_t v2y = vertex_y[vi2];
          real_t v2z = vertex_z[vi2];
          real_t v3x = vertex_x[vi3];
          real_t v3y = vertex_y[vi3];
          real_t v3z = vertex_z[vi3];

          // There are two candidates to split the quad into two triangles.
          //
          // Choose the shortest edge.
          // TODO: Is it better to determine the edge to split by calculating
          // the area of each triangle?
          //
          // +---+
          // |\  |
          // | \ |
          // |  \|
          // +---+
          //
          // +---+
          // |  /|
          // | / |
          // |/  |
          // +---+

          real_t e02x = v2x - v0x;
          real_t e02y = v2y - v0y;
          real_t e02z = v2z - v0z;
          real_t e13x = v3x - v1x;
          real_t e13y = v3y - v1y;
          real_t e13z = v3z - v1z;

          real_t sqr02 = e02x * e02x + e02y * e02y + e02z * e02z;
          real_t sqr13 = e13x * e13x + e13y * e13y + e13z * e13z;

          index_t idx0 = to_index(i0);
          index_t idx1 = to_index(i1);
          index_t idx2 = to_index(i2);
          index_t idx3 = to_index(i3);

          if (sqr02 < sqr13) {
            // [0, 1, 2], [0, 2, 3]
            shape->mesh.indices.push_back(idx0);
            shape->mesh.indices.push_back(idx1);
            shape->mesh.indices.push_back(idx2);

            shape->mesh.indices.push_back(idx0);
            shape->mesh.indices.push_back(idx2);
            shape->mesh.indices.push_back(idx3);
          } else {
            // [0, 1, 3], [1, 2, 3]
            shape->mesh.indices.push_back(idx0);
            shape->mesh.indices.push_back(idx1);
            shape->mesh.indices.push_back(idx3);

            shape->mesh.indices.push_back(idx1);
            shape->mesh.indices.push_back(idx2);
            shape->mesh.indices.push_back(idx3);
          }

          // Two triangle faces
          shape->mesh.num_face_vertices.push_back(3);
          shape->mesh.num_face_vertices.push_back(3);

          shape->mesh.smoothing_group_ids.push_back(face.smoothing_group_id);
          shape->mesh.smoothing_group_ids.push_back(face.smoothing_group_id);

        } else {
          vertex_index_t i0 = face.vertex_indices[0];
          vertex_index_t i1(-1);
          vertex_index_t i2 = face.vertex_indices[1];

          // find the two axes to work in
          size_t axes[2] = {1, 2};
          for (size_t k = 0; k < npolys; ++k) {
            i0 = face.vertex_indices[(k + 0) % npolys];
            i1 = face.vertex_indices[(k + 1) % npolys];
            i2 = face.vertex_indices[(k + 2) % npolys];
            size_t vi0 = size_t(i0.v_idx);
            size_t vi1 = size_t(i1.v_idx);
            size_t vi2 = size_t(i2.v_idx);

            if ((vi0 >= vertex_x.size()) || (vi1 >= vertex_x.size()) || (vi2 >= vertex_x.size())) {
              // Invalid triangle.
              // FIXME(syoyo): Is it ok to simply skip this invalid triangle?
              continue;
            }
            real_t v0x = vertex_x[vi0];
            real_t v0y = vertex_y[vi0];
            real_t v0z = vertex_z[vi0];
            real_t v1x = vertex_x[vi1];
            real_t v1y = vertex_y[vi1];
            real_t v1z = vertex_z[vi1];
            real_t v2x = vertex_x[vi2];
            real_t v2y = vertex_y[vi2];
            real_t v2z = vertex_z[vi2];
            real_t e0x = v1x - v0x;
            real_t e0y = v1y - v0y;
            real_t e0z = v1z - v0z;
            real_t e1x = v2x - v1x;
            real_t e1y = v2y - v1y;
            real_t e1z = v2z - v1z;
            real_t cx = std::fabs(e0y * e1z - e0z * e1y);
            real_t cy = std::fabs(e0z * e1x - e0x * e1z);
            real_t cz = std::fabs(e0x * e1y - e0y * e1x);
            constexpr real_t epsilon = std::numeric_limits<real_t>::epsilon();
            // std::cout << "cx " << cx << ", cy " << cy << ", cz " << cz <<
            // "\n";
            if (cx > epsilon || cy > epsilon || cz > epsilon) {
              // std::cout << "corner\n";
              // found a corner
              if (cx > cy && cx > cz) {
                // std::cout << "pattern0\n";
              } else {
                // std::cout << "axes[0] = 0\n";
                axes[0] = 0;
                if (cz > cx && cz > cy) {
                  // std::cout << "axes[1] = 1\n";
                  axes[1] = 1;
                }
              }
              break;
            }
          }

          face_t remainingFace = face;  // copy
          size_t guess_vert = 0;
          vertex_index_t ind[3];
          real_t vx[3];
          real_t vy[3];

          // How many iterations can we do without decreasing the remaining
          // vertices.
          size_t remainingIterations = face.vertex_indices.size();
          size_t previousRemainingVertices = remainingFace.vertex_indices.size();

          while (remainingFace.vertex_indices.size() > 3 && remainingIterations > 0) {
            // std::cout << "remainingIterations " << remainingIterations <<
            // "\n";

            npolys = remainingFace.vertex_indices.size();
            if (guess_vert >= npolys) {
              guess_vert -= npolys;
            }

            if (previousRemainingVertices != npolys) {
              // The number of remaining vertices decreased. Reset counters.
              previousRemainingVertices = npolys;
              remainingIterations = npolys;
            } else {
              // We didn't consume a vertex on previous iteration, reduce the
              // available iterations.
              remainingIterations--;
            }

            for (size_t k = 0; k < 3; k++) {
              ind[k] = remainingFace.vertex_indices[(guess_vert + k) % npolys];
              size_t vi = size_t(ind[k].v_idx);
              if ((vi >= vertex_x.size())) {
                // ???
                vx[k] = static_cast<real_t>(0.0);
                vy[k] = static_cast<real_t>(0.0);
              } else {
                vx[k] = (axes[0] == 0 ? vertex_x[vi] : (axes[0] == 1 ? vertex_y[vi] : vertex_z[vi]));
                vy[k] = (axes[1] == 0 ? vertex_x[vi] : (axes[1] == 1 ? vertex_y[vi] : vertex_z[vi]));
              }
            }

            //
            // area is calculated per face
            //
            real_t e0x = vx[1] - vx[0];
            real_t e0y = vy[1] - vy[0];
            real_t e1x = vx[2] - vx[1];
            real_t e1y = vy[2] - vy[1];
            real_t cross = e0x * e1y - e0y * e1x;
            // std::cout << "axes = " << axes[0] << ", " << axes[1] << "\n";
            // std::cout << "e0x, e0y, e1x, e1y " << e0x << ", " << e0y << ", "
            // << e1x << ", " << e1y << "\n";

            real_t area = (vx[0] * vy[1] - vy[0] * vx[1]) * static_cast<real_t>(0.5);
            // std::cout << "cross " << cross << ", area " << area << "\n";
            // if an internal angle
            if (cross * area < static_cast<real_t>(0.0)) {
              // std::cout << "internal \n";
              guess_vert += 1;
              // std::cout << "guess vert : " << guess_vert << "\n";
              continue;
            }

            // check all other verts in case they are inside this triangle
            bool overlap = false;
            for (size_t otherVert = 3; otherVert < npolys; ++otherVert) {
              size_t idx = (guess_vert + otherVert) % npolys;

              if (idx >= remainingFace.vertex_indices.size()) {
                // std::cout << "???0\n";
                // ???
                continue;
              }

              size_t ovi = size_t(remainingFace.vertex_indices[idx].v_idx);

              if ((ovi >= vertex_x.size())) {
                // std::cout << "???1\n";
                // ???
                continue;
              }
              real_t tx = (axes[0] == 0 ? vertex_x[ovi] : (axes[0] == 1 ? vertex_y[ovi] : vertex_z[ovi]));
              real_t ty = (axes[1] == 0 ? vertex_x[ovi] : (axes[1] == 1 ? vertex_y[ovi] : vertex_z[ovi]));
              if (pnpoly(3, vx, vy, tx, ty)) {
                // std::cout << "overlap\n";
                overlap = true;
                break;
              }
            }

            if (overlap) {
              // std::cout << "overlap2\n";
              guess_vert += 1;
              continue;
            }

            // this triangle is an ear
            {
              index_t idx0, idx1, idx2;
              idx0 = to_index(ind[0]);
              idx1 = to_index(ind[1]);
              idx2 = to_index(ind[2]);

              shape->mesh.indices.push_back(idx0);
              shape->mesh.indices.push_back(idx1);
              shape->mesh.indices.push_back(idx2);

              shape->mesh.num_face_vertices.push_back(3);
              shape->mesh.smoothing_group_ids.push_back(face.smoothing_group_id);
            }

            // remove v1 from the list
            size_t removed_vert_index = (guess_vert + 1) % npolys;
            while (removed_vert_index + 1 < npolys) {
              remainingFace.vertex_indices[removed_vert_index] = remainingFace.vertex_indices[removed_vert_index + 1];
              removed_vert_index += 1;
            }
            remainingFace.vertex_indices.pop_back();
          }

          // std::cout << "remainingFace.vi.size = " <<
          // remainingFace.vertex_indices.size() << "\n";
          if (remainingFace.vertex_indices.size() == 3) {
            i0 = remainingFace.vertex_indices[0];
            i1 = remainingFace.vertex_indices[1];
            i2 = remainingFace.vertex_indices[2];
            {
              index_t idx0, idx1, idx2;
              idx0 = to_index(i0);
              idx1 = to_index(i1);
              idx2 = to_index(i2);

              shape->mesh.indices.push_back(idx0);
              shape->mesh.indices.push_back(idx1);
              shape->mesh.indices.push_back(idx2);

              shape->mesh.num_face_vertices.push_back(3);
              shape->mesh.smoothing_group_ids.push_back(face.smoothing_group_id);
            }
          }
        }  // npolys
      } else {
        for (size_t k = 0; k < npolys; k++) {
          index_t idx;
          idx = to_index(face.vertex_indices[k]);
          shape->mesh.indices.push_back(idx);
        }

        shape->mesh.num_face_vertices.push_back(static_cast<unsigned char>(npolys));
        shape->mesh.smoothing_group_ids.push_back(face.smoothing_group_id);  // per face
      }
    }
  }

  return true;
}

// Split a string with specified delimiter character and escape character.
// https://rosettacode.org/wiki/Tokenize_a_string_with_escaping#C.2B.2B
static void SplitString(const std::string& s, char delim, char escape, std::vector<std::string>& elems) {
  std::string token;

  bool escaping = false;
  for (size_t i = 0; i < s.size(); ++i) {
    char ch = s[i];
    if (escaping) {
      escaping = false;
    } else if (ch == escape) {
      escaping = true;
      continue;
    } else if (ch == delim) {
      if (!token.empty()) {
        elems.push_back(token);
      }
      token.clear();
      continue;
    }
    token += ch;
  }

  elems.push_back(token);
}

static std::string JoinPath(const std::string& dir, const std::string& filename) {
  if (dir.empty()) {
    return filename;
  } else {
    // check '/'
    char lastChar = *dir.rbegin();
    if (lastChar != '/') {
      return dir + std::string("/") + filename;
    } else {
      return dir + filename;
    }
  }
}

void LoadMtl(std::map<std::string, int>* material_map, std::vector<material_t>* materials, std::istream* inStream, std::string* warning, std::string* err) {
  (void)err;

  material_t material;
  InitMaterial(&material);

  std::stringstream warn_ss;

  size_t line_no = 0;
  std::string linebuf;
  while (inStream->peek() != -1) {
    std::getline(*inStream, linebuf);
    line_no++;

    // Trim trailing whitespace.
    if (linebuf.size() > 0) {
      linebuf = linebuf.substr(0, linebuf.find_last_not_of(" \t") + 1);
    }

    // Trim newline '\r\n' or '\n'
    if (linebuf.size() > 0) {
      if (linebuf[linebuf.size() - 1] == '\n')
        linebuf.erase(linebuf.size() - 1);
    }
    if (linebuf.size() > 0) {
      if (linebuf[linebuf.size() - 1] == '\r')
        linebuf.erase(linebuf.size() - 1);
    }

    // Skip if empty line.
    if (linebuf.empty()) {
      continue;
    }

    // Skip leading space.
    const char* token = linebuf.c_str();
    token += strspn(token, " \t");

    assert(token);
    if (token[0] == '\0')
      continue;  // empty line

    if (token[0] == '#')
      continue;  // comment line

    // new mtl
    if ((0 == _strnicmp(token, "newmtl", 6)) && IS_SPACE((token[6]))) {
      // flush previous material.
      if (!material.name.empty()) {
        std::transform(material.name.begin(), material.name.end(), material.name.begin(), tolower);
        material_map->insert(std::pair<std::string, int>(material.name, static_cast<int>(materials->size())));
        materials->push_back(material);
      }

      // initial temporary material
      InitMaterial(&material);

      // set new mtl name
      token += 7;
      {
        std::stringstream sstr;
        sstr << token;
        material.name = sstr.str();
      }
      continue;
    }

    // unknown parameter
    const char* _space = strchr(token, ' ');
    if (!_space) {
      _space = strchr(token, '\t');
    }

    std::ptrdiff_t len = _space ? _space - token : strlen(token);
    std::string key(token, static_cast<size_t>(len));
    std::string value = _space ? _space + 1 : "";
    material.unknown_parameter.emplace_back(std::pair<std::string, std::string>(key, value));
  }

  // flush last material.
  std::transform(material.name.begin(), material.name.end(), material.name.begin(), tolower);
  material_map->insert(std::pair<std::string, int>(material.name, static_cast<int>(materials->size())));
  materials->push_back(material);

  if (warning) {
    (*warning) = warn_ss.str();
  }
}

bool MaterialFileReader::operator()(const std::string& matId, std::vector<material_t>* materials, std::map<std::string, int>* matMap, std::string* warn, std::string* err) {
  if (!m_mtlBaseDir.empty()) {
#ifdef _WIN32
    char sep = ';';
#else
    char sep = ':';
#endif

    // https://stackoverflow.com/questions/5167625/splitting-a-c-stdstring-using-tokens-e-g
    std::vector<std::string> paths;
    paths.emplace_back();

    std::istringstream f(m_mtlBaseDir);
    std::string s;
    while (getline(f, s, sep)) {
      paths.push_back(s);
    }

    for (size_t i = 0; i < paths.size(); i++) {
      std::string filepath = JoinPath(paths[i], matId);
      std::ifstream matIStream(filepath.c_str());
      if (matIStream) {
        LoadMtl(matMap, materials, &matIStream, warn, err);
        return true;
      }
    }

    std::stringstream ss;
    ss << "Material file [ " << matId << " ] not found in a path : " << m_mtlBaseDir << "\n";
    if (warn) {
      (*warn) += ss.str();
    }
    return false;

  } else {
    std::string filepath = matId;
    std::ifstream matIStream(filepath.c_str());
    if (matIStream) {
      LoadMtl(matMap, materials, &matIStream, warn, err);

      return true;
    }

    std::stringstream ss;
    ss << "Material file [ " << filepath << " ] not found in a path : " << m_mtlBaseDir << "\n";
    if (warn) {
      (*warn) += ss.str();
    }

    return false;
  }
}

bool LoadObj(attrib_t* attrib, std::vector<shape_t>* shapes, std::vector<material_t>* materials, std::string* warn, std::string* err, const char* filename, const char* mtl_basedir,
  const char* mtl_custom_file, bool triangulate, bool default_vcols_fallback) {
  attrib->vertex_x.clear();
  attrib->vertex_y.clear();
  attrib->vertex_z.clear();
  attrib->normal_x.clear();
  attrib->normal_y.clear();
  attrib->normal_z.clear();
  attrib->texcoord_u.clear();
  attrib->texcoord_v.clear();
  attrib->texcoord_ws.clear();
  attrib->colors.clear();
  shapes->clear();

  std::stringstream errss;

  std::ifstream ifs(filename);
  if (!ifs) {
    errss << "Cannot open file [" << filename << "]\n";
    if (err) {
      (*err) = errss.str();
    }
    return false;
  }

  std::string baseDir = mtl_basedir ? mtl_basedir : "";
  if (baseDir.empty() == false) {
#ifndef _WIN32
    const char dirsep = '/';
    for (auto& c : baseDir) {
      if (c == '\\')
        c = '/';
    }
#else
    const char dirsep = '\\';
    for (auto& c : baseDir) {
      if (c == '/')
        c = '\\';
    }
#endif

    if (baseDir[baseDir.length() - 1] != dirsep)
      baseDir += dirsep;
  }

  MaterialFileReader matFileReader(baseDir);
  return LoadObj(attrib, shapes, materials, warn, err, &ifs, &matFileReader, mtl_custom_file, triangulate, default_vcols_fallback);
}

bool LoadObj(attrib_t* attrib, std::vector<shape_t>* shapes, std::vector<material_t>* materials, std::string* warn, std::string* err, std::istream* inStream,
  MaterialReader* readMatFn, const char* customMaterials, bool triangulate, bool default_vcols_fallback) {
  auto t0 = std::chrono::steady_clock::now();
  auto t_setup = std::chrono::steady_clock::now();
  std::stringstream errss;

  std::vector<real_t> vx, vy, vz;
  std::vector<real_t> vnx, vny, vnz;
  std::vector<real_t> vtu, vtv;
  std::vector<real_t> vtw;
  std::vector<real_t> vc;
  PrimGroup prim_group;
  std::string name;

  vx.reserve(65536);
  vy.reserve(65536);
  vz.reserve(65536);
  vnx.reserve(65536);
  vny.reserve(65536);
  vnz.reserve(65536);
  vtu.reserve(65536);
  vtv.reserve(65536);
  vtw.reserve(65536);
  vc.reserve(65536);

  // material
  std::map<std::string, int> material_map;
  int material = -1;

  // smoothing group id
  unsigned int current_smoothing_id = 0;  // Initial value. 0 means no smoothing.

  int greatest_v_idx = -1;
  int greatest_vn_idx = -1;
  int greatest_vt_idx = -1;

  shape_t shape;

  bool found_all_colors = true;

  size_t line_num = 0;
  size_t vertex_count = 0;
  size_t normal_count = 0;
  size_t texcoord_count = 0;
  size_t face_count = 0;
  size_t material_load_count = 0;
  double total_line_processing_time = 0.0;
  double vertex_parsing_time = 0.0;
  double normal_parsing_time = 0.0;
  double texcoord_parsing_time = 0.0;
  double face_parsing_time = 0.0;
  double material_parsing_time = 0.0;
  double other_tokens_time = 0.0;  // g, o, s, and other tokens
  double other_parsing_time = 0.0;

  BufferedReader reader(inStream);
  BufferedReader::Line line_view;

  auto t_setup_end = std::chrono::steady_clock::now();
  auto t_parsing_start = std::chrono::steady_clock::now();

  while (reader.getline(line_view)) {
    auto line_start = std::chrono::steady_clock::now();
    line_num++;

    // Skip if empty line.
    if (line_view.empty()) {
      auto line_end = std::chrono::steady_clock::now();
      total_line_processing_time += std::chrono::duration<double, std::milli>(line_end - line_start).count();
      continue;
    }

    char* token = line_view.begin;

    if (token[0] == '#') {
      auto line_end = std::chrono::steady_clock::now();
      total_line_processing_time += std::chrono::duration<double, std::milli>(line_end - line_start).count();
      continue;  // comment line
    }

    // Line processing setup complete (reading, copying, comment checks)
    auto token_processing_start = std::chrono::steady_clock::now();

    // Token processing overhead (branching logic)
    auto token_start = std::chrono::steady_clock::now();

    // vertex
    if (token[0] == 'v' && IS_SPACE(token[1])) {
      auto t_start = std::chrono::steady_clock::now();
      token += 2;
      real_t x, y, z;
      real_t r, g, b;

      found_all_colors &= parseVertexWithColor(&x, &y, &z, &r, &g, &b, &token);

      vx.push_back(x);
      vy.push_back(y);
      vz.push_back(z);
      vertex_count++;

      if (found_all_colors || default_vcols_fallback) {
        vc.push_back(r);
        vc.push_back(g);
        vc.push_back(b);
      }
      auto t_end = std::chrono::steady_clock::now();
      vertex_parsing_time += std::chrono::duration<double, std::milli>(t_end - t_start).count();
      auto line_end = std::chrono::steady_clock::now();
      total_line_processing_time += std::chrono::duration<double, std::milli>(line_end - line_start).count();
      continue;
    }

    // normal
    if (token[0] == 'v' && token[1] == 'n' && IS_SPACE((token[2]))) {
      auto t_start = std::chrono::steady_clock::now();
      token += 3;
      real_t x, y, z;
      parseReal3(&x, &y, &z, &token);
      vnx.push_back(x);
      vny.push_back(y);
      vnz.push_back(z);
      normal_count++;
      auto t_end = std::chrono::steady_clock::now();
      normal_parsing_time += std::chrono::duration<double, std::milli>(t_end - t_start).count();
      auto line_end = std::chrono::steady_clock::now();
      total_line_processing_time += std::chrono::duration<double, std::milli>(line_end - line_start).count();
      continue;
    }

    // texcoord
    if (token[0] == 'v' && token[1] == 't' && IS_SPACE((token[2]))) {
      auto t_start = std::chrono::steady_clock::now();
      token += 3;
      real_t x, y;
      parseReal2(&x, &y, &token);
      vtu.push_back(x);
      vtv.push_back(y);
      texcoord_count++;
      auto t_end = std::chrono::steady_clock::now();
      texcoord_parsing_time += std::chrono::duration<double, std::milli>(t_end - t_start).count();
      auto line_end = std::chrono::steady_clock::now();
      total_line_processing_time += std::chrono::duration<double, std::milli>(line_end - line_start).count();
      continue;
    }

    // face
    if (token[0] == 'f' && IS_SPACE(token[1])) {
      auto t_start = std::chrono::steady_clock::now();
      token += 2;
      token += strspn(token, " \t");

      face_t face;

      face.smoothing_group_id = current_smoothing_id;
      face.vertex_indices.reserve(3);

      while (IS_NEW_LINE(token[0]) == false) {
        vertex_index_t vi;
        if (!parseTriple(&token, static_cast<int>(vx.size()), static_cast<int>(vnx.size()), static_cast<int>(vtu.size()), &vi)) {
          if (err) {
            std::stringstream ss;
            ss << "Failed parse `f' line(e.g. zero value for face index. line " << line_num << ".)\n";
            (*err) += ss.str();
          }
          return false;
        }

        greatest_v_idx = greatest_v_idx > vi.v_idx ? greatest_v_idx : vi.v_idx;
        greatest_vn_idx = greatest_vn_idx > vi.vn_idx ? greatest_vn_idx : vi.vn_idx;
        greatest_vt_idx = greatest_vt_idx > vi.vt_idx ? greatest_vt_idx : vi.vt_idx;

        face.vertex_indices.push_back(vi);
        size_t n = strspn(token, " \t\r");
        token += n;
      }

      // replace with emplace_back + std::move on C++11
      prim_group.faceGroup.push_back(face);
      face_count++;
      auto t_end = std::chrono::steady_clock::now();
      face_parsing_time += std::chrono::duration<double, std::milli>(t_end - t_start).count();
      auto line_end = std::chrono::steady_clock::now();
      total_line_processing_time += std::chrono::duration<double, std::milli>(line_end - line_start).count();

      continue;
    }

    // use mtl
    if ((0 == _strnicmp(token, "usemtl", 6))) {
      auto t_start = std::chrono::steady_clock::now();
      token += 6;
      std::string namebuf = parseString(&token);
      std::transform(namebuf.begin(), namebuf.end(), namebuf.begin(), tolower);
      int newMaterialId = -1;
      std::map<std::string, int>::const_iterator it = material_map.find(namebuf);
      if (it != material_map.end()) {
        newMaterialId = it->second;
      } else {
        // { error!! material not found }
        if (warn) {
          (*warn) += "material [ '" + namebuf + "' ] not found in .mtl\n";
        }
      }

      if (newMaterialId != material) {
        // Create per-face material. Thus we don't add `shape` to `shapes` at this time.
        // just clear `faceGroup` after `exportGroupsToShape()` call.
        exportGroupsToShape(&shape, prim_group, name, triangulate, vx, vy, vz, material, warn);
        prim_group.faceGroup.clear();
        material = newMaterialId;
      }
      auto t_end = std::chrono::steady_clock::now();
      material_parsing_time += std::chrono::duration<double, std::milli>(t_end - t_start).count();
      auto line_end = std::chrono::steady_clock::now();
      total_line_processing_time += std::chrono::duration<double, std::milli>(line_end - line_start).count();

      continue;
    }

    // load mtl
    if ((0 == _strnicmp(token, "mtllib", 6)) && IS_SPACE((token[6]))) {
      if (readMatFn) {
        token += 7;

        std::vector<std::string> filenames;
        if ((customMaterials == nullptr) || (customMaterials[0] == 0)) {
          SplitString(std::string(token), ' ', '\\', filenames);
        } else {
          filenames.emplace_back(customMaterials);
        }

        if (filenames.empty()) {
          if (warn) {
            std::stringstream ss;
            ss << "Looks like empty filename for mtllib. Use default "
                  "material (line "
               << line_num << ".)\n";

            (*warn) += ss.str();
          }
        } else {
          auto t_material_start = std::chrono::steady_clock::now();
          bool found = false;
          for (size_t s = 0; s < filenames.size(); s++) {
            std::string warn_mtl;
            std::string err_mtl;
            bool ok = (*readMatFn)(filenames[s].c_str(), materials, &material_map, &warn_mtl, &err_mtl);
            if (warn && (!warn_mtl.empty())) {
              (*warn) += warn_mtl;
            }

            if (err && (!err_mtl.empty())) {
              (*err) += err_mtl;
            }

            if (ok) {
              found = true;
              material_load_count++;
              break;
            }
          }

          if (!found) {
            if (warn) {
              (*warn) +=
                "Failed to load material file(s). Use default "
                "material.\n";
            }
          }
        }
      }

      continue;
    }

    // group name
    if (token[0] == 'g' && IS_SPACE(token[1])) {
      auto t_start = std::chrono::steady_clock::now();
      // flush previous face group.
      bool ret = exportGroupsToShape(&shape, prim_group, name, triangulate, vx, vy, vz, material, warn);
      (void)ret;  // return value not used.

      if (shape.mesh.indices.size() > 0) {
        shapes->push_back(shape);
      }

      shape = shape_t();

      // material = -1;
      prim_group.clear();

      std::vector<std::string> names;

      while (!IS_NEW_LINE(token[0])) {
        std::string str = parseString(&token);
        names.push_back(str);
        token += strspn(token, " \t\r");  // skip tag
      }

      // names[0] must be 'g'

      if (names.size() < 2) {
        // 'g' with empty names
        if (warn) {
          std::stringstream ss;
          ss << "Empty group name. line: " << line_num << "\n";
          (*warn) += ss.str();
          name = "";
        }
      } else {
        std::stringstream ss;
        ss << names[1];
        for (size_t i = 2; i < names.size(); i++) {
          ss << " " << names[i];
        }
        name = ss.str();
      }
      auto t_end = std::chrono::steady_clock::now();
      other_tokens_time += std::chrono::duration<double, std::milli>(t_end - t_start).count();
      auto line_end = std::chrono::steady_clock::now();
      total_line_processing_time += std::chrono::duration<double, std::milli>(line_end - line_start).count();
      continue;
    }

    // object name
    if (token[0] == 'o' && IS_SPACE(token[1])) {
      auto t_start = std::chrono::steady_clock::now();
      // flush previous face group.
      bool ret = exportGroupsToShape(&shape, prim_group, name, triangulate, vx, vy, vz, material, warn);
      (void)ret;  // return value not used.

      if (shape.mesh.indices.size() > 0) {
        shapes->push_back(shape);
      }

      // material = -1;
      prim_group.clear();
      shape = shape_t();

      // @todo { multiple object name? }
      token += 2;
      std::stringstream ss;
      ss << token;
      name = ss.str();
      auto t_end = std::chrono::steady_clock::now();
      other_tokens_time += std::chrono::duration<double, std::milli>(t_end - t_start).count();
      auto line_end = std::chrono::steady_clock::now();
      total_line_processing_time += std::chrono::duration<double, std::milli>(line_end - line_start).count();

      continue;
    }

    if (token[0] == 's' && IS_SPACE(token[1])) {
      auto t_start = std::chrono::steady_clock::now();
      // smoothing group id
      token += 2;

      // skip space.
      token += strspn(token, " \t");  // skip space

      if (token[0] == '\0') {
        auto t_end = std::chrono::steady_clock::now();
        other_tokens_time += std::chrono::duration<double, std::milli>(t_end - t_start).count();
        auto line_end = std::chrono::steady_clock::now();
        total_line_processing_time += std::chrono::duration<double, std::milli>(line_end - line_start).count();
        continue;
      }

      if (token[0] == '\r' || token[1] == '\n') {
        auto t_end = std::chrono::steady_clock::now();
        other_tokens_time += std::chrono::duration<double, std::milli>(t_end - t_start).count();
        auto line_end = std::chrono::steady_clock::now();
        total_line_processing_time += std::chrono::duration<double, std::milli>(line_end - line_start).count();
        continue;
      }

      if (strlen(token) >= 3 && token[0] == 'o' && token[1] == 'f' && token[2] == 'f') {
        current_smoothing_id = 0;
      } else {
        // assume number
        int smGroupId = parseInt(&token);
        if (smGroupId < 0) {
          // parse error. force set to 0.
          // FIXME(syoyo): Report warning.
          current_smoothing_id = 0;
        } else {
          current_smoothing_id = static_cast<unsigned int>(smGroupId);
        }
      }
      auto t_end = std::chrono::steady_clock::now();
      other_tokens_time += std::chrono::duration<double, std::milli>(t_end - t_start).count();
      auto line_end = std::chrono::steady_clock::now();
      total_line_processing_time += std::chrono::duration<double, std::milli>(line_end - line_start).count();

      continue;
    }
  }

  auto t_parsing_end = std::chrono::steady_clock::now();
  auto t_triangulation_start = std::chrono::steady_clock::now();

  // not all vertices have colors, no default colors desired? -> clear colors
  if (!found_all_colors && !default_vcols_fallback) {
    vc.clear();
  }

  if (greatest_v_idx >= static_cast<int>(vx.size())) {
    if (warn) {
      std::stringstream ss;
      ss << "Vertex indices out of bounds (line " << line_num << ".)\n\n";
      (*warn) += ss.str();
    }
  }
  if (greatest_vn_idx >= static_cast<int>(vnx.size())) {
    if (warn) {
      std::stringstream ss;
      ss << "Vertex normal indices out of bounds (line " << line_num << ".)\n\n";
      (*warn) += ss.str();
    }
  }
  if (greatest_vt_idx >= static_cast<int>(vtu.size())) {
    if (warn) {
      std::stringstream ss;
      ss << "Vertex texcoord indices out of bounds (line " << line_num << ".)\n\n";
      (*warn) += ss.str();
    }
  }

  bool ret = exportGroupsToShape(&shape, prim_group, name, triangulate, vx, vy, vz, material, warn);
  if (ret || shape.mesh.indices.size()) {
    shapes->push_back(shape);
  }
  prim_group.clear();  // for safety

  if (err) {
    (*err) += errss.str();
  }

  attrib->vertex_x.swap(vx);
  attrib->vertex_y.swap(vy);
  attrib->vertex_z.swap(vz);
  attrib->normal_x.swap(vnx);
  attrib->normal_y.swap(vny);
  attrib->normal_z.swap(vnz);
  attrib->texcoord_u.swap(vtu);
  attrib->texcoord_v.swap(vtv);
  attrib->texcoord_ws.swap(vtw);
  attrib->colors.swap(vc);

  auto t_end = std::chrono::steady_clock::now();

  auto setup_time = (t_setup_end - t0).count() / 1.0e+6;
  auto parsing_time = (t_parsing_end - t_parsing_start).count() / 1.0e+6;
  auto triangulation_time = (t_end - t_triangulation_start).count() / 1.0e+6;
  auto total_time = (t_end - t0).count() / 1.0e+6;

  double individual_timings = vertex_parsing_time + normal_parsing_time + texcoord_parsing_time + face_parsing_time + material_parsing_time + other_parsing_time;
  double line_overhead_time = total_line_processing_time - individual_timings;
  std::stringstream perf_ss;
  perf_ss << "\n=== OBJ Loader Performance Statistics ===\n";
  perf_ss << "Total time: " << total_time << " ms\n";
  perf_ss << "Setup time: " << setup_time << " ms (" << (setup_time / total_time * 100) << "%)\n";
  perf_ss << "Parsing time: " << parsing_time << " ms (" << (parsing_time / total_time * 100) << "%)\n";
  perf_ss << "  ├── Line overhead: " << line_overhead_time << " ms (" << (line_overhead_time / parsing_time * 100) << "%)\n";
  perf_ss << "  ├── Vertex parsing: " << vertex_parsing_time << " ms (" << (vertex_parsing_time / parsing_time * 100) << "%)\n";
  perf_ss << "  ├── Normal parsing: " << normal_parsing_time << " ms (" << (normal_parsing_time / parsing_time * 100) << "%)\n";
  perf_ss << "  ├── Texcoord parsing: " << texcoord_parsing_time << " ms (" << (texcoord_parsing_time / parsing_time * 100) << "%)\n";
  perf_ss << "  ├── Face parsing: " << face_parsing_time << " ms (" << (face_parsing_time / parsing_time * 100) << "%)\n";
  perf_ss << "  ├── Material parsing: " << material_parsing_time << " ms (" << (material_parsing_time / parsing_time * 100) << "%)\n";
  perf_ss << "  ├── Other tokens: " << other_tokens_time << " ms (" << (other_tokens_time / parsing_time * 100) << "%)\n";
  perf_ss << "  └── Other parsing: " << other_parsing_time << " ms (" << (other_parsing_time / parsing_time * 100) << "%)\n";
  perf_ss << "Triangulation time: " << triangulation_time << " ms (" << (triangulation_time / total_time * 100) << "%)\n";
  perf_ss << "\nData statistics:\n";
  perf_ss << "Lines processed: " << line_num << "\n";
  perf_ss << "Vertices: " << vertex_count << "\n";
  perf_ss << "Normals: " << normal_count << "\n";
  perf_ss << "Texcoords: " << texcoord_count << "\n";
  perf_ss << "Faces: " << face_count << "\n";
  perf_ss << "Material files loaded: " << material_load_count << "\n";
  perf_ss << "=======================================\n";

  if (warn) {
    (*warn) += perf_ss.str();
  } else {
    std::cerr << perf_ss.str();
  }

  return true;
}

#ifdef __clang__
# pragma clang diagnostic pop
#endif
}  // namespace tinyobj

#endif
