// win_compat.h — minimal Windows shims for src/llama.cpp's remaining POSIX-isms
// (sysconf + the advisory posix_madvise prefetch hint). The canonical
// common/profiler.cpp is already #ifdef _WIN32-aware; this covers the one file the
// authors left unguarded so prima.cpp builds with mingw/clang. MIT (matches
// prima.cpp). Active only on _WIN32.
#pragma once
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstddef>

#ifndef _SC_PAGESIZE
#define _SC_PAGESIZE 30
#endif
#ifndef _SC_NPROCESSORS_ONLN
#define _SC_NPROCESSORS_ONLN 58
#endif

// sysconf: only the two queries prima.cpp's llama.cpp uses.
static inline long sysconf(int name) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    switch (name) {
        case _SC_PAGESIZE:         return (long) si.dwPageSize;
        case _SC_NPROCESSORS_ONLN: return (long) si.dwNumberOfProcessors;
        default:                   return -1L;
    }
}

// posix_madvise is an advisory prefetch hint; a no-op on Windows is correct —
// the model still loads, we just skip that one optimization.
#ifndef POSIX_MADV_NORMAL
#define POSIX_MADV_NORMAL     0
#define POSIX_MADV_RANDOM     1
#define POSIX_MADV_SEQUENTIAL 2
#define POSIX_MADV_WILLNEED   3
#define POSIX_MADV_DONTNEED   4
#endif

static inline int posix_madvise(void * addr, size_t len, int advice) {
    (void) addr; (void) len; (void) advice;
    return 0;
}
#endif // _WIN32
