# bcdec
Small header-only C library to decompress any BC compressed image inspired by incredible stb libaries (<http://nothings.org/stb>)

Written by Sergii *"iOrange"* Kudlai in 2022.

This library provides functions to decompress blocks of BC compressed images.  

This library does not allocate memory and is trying to use as less stack as possible.

The library was never optimized specifically for speed but for the overall size.  
It has zero external dependencies and is not using any runtime functions.

### Supported BC formats:
- BC1 (also known as DXT1) + it's "binary alpha" variant BC1A (DXT1A)
- BC2 (also known as DXT3)
- BC3 (also known as DXT5)
- BC4 (also known as ATI1N)
- BC5 (also known as ATI2N)
- BC6H (HDR format)
- BC7

---

BC1/BC2/BC3/BC7 are expected to decompress into 4x4 RGBA blocks 8bit per component (32bit pixel)  
BC4/BC5 are expected to decompress into 4x4 R/RG blocks 8bit per component (8bit and 16bit pixel)  
BC6H is expected to decompress into 4x4 RGB blocks of either 32bit float or 16bit "half" per
component (96bit or 48bit pixel)  

*\* Note that BCDEC\_BC4BC5\_PRECISE option enables signed/unsigned and float support for BC4/BC5*

---

You will also find included test program that converts compressed DDS files into TGA/HDR.  
It is a good start to learn on how to use the **bcdec** library.  
It comes with some test images in the *test_images* folder and a batch script *test_bcdec.bat* to run over them.

---

Used HDRI image "Lythwood Room" from <https://polyhaven.com/a/lythwood_room> licensed under CC0 license.

---

### CREDITS:
 - Aras Pranckevičius (@aras-p)
     - BC1/BC3 decoders optimizations (up to 3x the speed)
     - BC6H/BC7 bits pulling routines optimizations
     - optimized BC6H by moving unquantize out of the loop
     - Split BC6H decompression function into 'half' and 'float' variants
  
 - Michael Schmidt (@RunDevelopment)
     - Found better "magic" coefficients for integer interpolation
       of reference colors in BC1 color block, that match with
       the floating point interpolation. This also made it faster
       than integer division.

