#include "SSSE3Helper.h"

__m128i SSSE3Helper::shufflePseudo(unsigned char (&a)[16], unsigned char (&b)[16]){
	unsigned char r[16];

	//INTEL
	for (int i = 0; i<16; i++){
		if (b[i] & 0x80)
			r[i] = 0;
		else
			r[i] = a[b[i] & 0x0F];
	}

	return _mm_loadu_si128((__m128i*)&r[0]);
};

__m128i SSSE3Helper::shuffleInstruction(unsigned char (&a)[16], unsigned char(&b)[16]){
	__m128i ma = _mm_loadu_si128((__m128i*)&a[0]);
	__m128i mb = _mm_loadu_si128((__m128i*)&b[0]);
	return _mm_shuffle_epi8(ma, mb);
};

__m128i SSSE3Helper::alignrPseudo(unsigned char (&a)[16], unsigned char (&b)[16], int n){
	//INTEL
	//t1[255:128] = a;
	//t1[127:0] = b;
	//t1[255:0] = t1[255:0] >> (8 * n); // unsigned shift
	//r[127:0] = t1[127:0];
	
	unsigned char t1[32];//an array of 32 bytes, totaling 256 bits

	for (int i = 0;i<16;i++){
		t1[i+16] = a[i];//put a in the second half of the array
		t1[i] = b[i];//put b in the first half of the array
	}

	for (int i = 0;i<32;i++){
		t1[i] = t1[i+n];//shift entire bytes down the array.
		//Effectivly the same as a bitshift where shifts may only be made in increments of 8
	}
	return _mm_loadu_si128((__m128i*)&t1[0]);
};

__m128i SSSE3Helper::alignrInstruction(unsigned char (&a)[16], unsigned char (&b)[16], int n){
	__m128i ma = _mm_loadu_si128((__m128i*)&a[0]);
	__m128i mb = _mm_loadu_si128((__m128i*)&b[0]);

	__m128i r = _mm_alignr_epi8(ma, mb, 0);

	return r;
};


__m128i SSSE3Helper::mulhrsPseudo( signed short (&a)[8], signed short (&b)[8]){
	signed short r[8];

	for (int i = 0; i < 8; i++) {
		r[i] =  (( (int)((a[i] * b[i]) >> 14) + 1) >> 1) & 0xFFFF;
	}

	return _mm_loadu_si128((__m128i*)&r[0]);
};


__m128i SSSE3Helper::mulhrsInstruction(signed short (&a)[8], signed short (&b)[8]){
	__m128i ma = _mm_loadu_si128((__m128i*)&a[0]);
	__m128i mb = _mm_loadu_si128((__m128i*)&b[0]);
	return _mm_mulhrs_epi16(ma, mb);
};

bool SSSE3Helper::equates(__m128i a, __m128i b){
	return _mm_test_all_ones(_mm_cmpeq_epi8(a,b));
}