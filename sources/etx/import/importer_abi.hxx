#pragma once

#include <stdint.h>

#if defined(__cplusplus)
# define ETX_IMPORT_EXTERN extern "C"
#else
# define ETX_IMPORT_EXTERN extern
#endif

#if defined(_WIN32)
# define ETX_IMPORT_CALL   __cdecl
# define ETX_IMPORT_EXPORT ETX_IMPORT_EXTERN __declspec(dllexport)
#else
# define ETX_IMPORT_CALL
# define ETX_IMPORT_EXPORT ETX_IMPORT_EXTERN __attribute__((visibility("default")))
#endif

// Paths and messages are UTF-8. All memory remains owned by its allocating module.
// Plugins stay loaded until every request has returned. Exceptions may not cross this boundary.
enum { ETX_IMPORT_ABI_VERSION = 1u, ETX_IMPORT_EXTERNAL_MATERIALS = 1u, ETX_IMPORT_DEPENDENCY = 2u };

typedef int32_t(ETX_IMPORT_CALL* EtxImportProgress)(void* context, uint32_t completed, uint32_t total, const char* stage);
typedef void(ETX_IMPORT_CALL* EtxImportDependency)(void* context, const char* path, uint32_t external_material_override);

typedef struct EtxImportRequest EtxImportRequest;
typedef struct EtxImporterApi EtxImporterApi;

struct EtxImportRequest {
  uint32_t size;
  const char* source_path;
  const char* output_directory;
  const char* runtime_directory;
  void* context;
  EtxImportProgress progress;
};

struct EtxImporterApi {
  uint32_t size;
  uint32_t abi_version;
  const char* id;
  const char* name;
  const char* extensions;
  int32_t(ETX_IMPORT_CALL* probe)(const char* source_path);
  // DEPENDENCY requests inspect an auxiliary file using the root importer, without probing its format.
  int32_t(ETX_IMPORT_CALL* inspect)(const char* source_path, uint32_t flags, void* context, EtxImportDependency dependency, char* error, uint32_t error_capacity);
  // Conversion returns an absolute native document path inside output_directory.
  int32_t(ETX_IMPORT_CALL* convert)(const EtxImportRequest* request, char* document_path, uint32_t path_capacity, char* error, uint32_t error_capacity);
};

typedef const EtxImporterApi*(ETX_IMPORT_CALL* EtxGetImporterApi)(uint32_t abi_version);
