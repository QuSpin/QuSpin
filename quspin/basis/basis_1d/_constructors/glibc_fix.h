#ifdef __linux__
    asm (".symver memcpy, memcpy@GLIBC_2.2.5");
#endif