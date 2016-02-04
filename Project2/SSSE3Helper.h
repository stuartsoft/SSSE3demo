#ifndef SSSE3HELPER_H
#define SSSE3HELPER_H

#include <stdlib.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <iostream>

class SSSE3Helper{

public:
	static __m128i shufflePseudo(unsigned char (&a)[16], unsigned char (&b)[16]);
	static __m128i shuffleInstruction(unsigned char (&a)[16], unsigned char(&b)[16]);

	static __m128i alignrPseudo(unsigned char (&a)[16], unsigned char (&b)[16], int n);
	static __m128i alignrInstruction(unsigned char (&a)[16], unsigned char (&b)[16], int n);


	static __m128i mulhrsPseudo( signed short (&a)[8], signed short (&b)[8]);
	static __m128i mulhrsInstruction(signed short (&a)[8], signed short (&b)[8]);

	static bool equates(__m128i a, __m128i b);
};

#endif