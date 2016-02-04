
#include "StopWatch.h"
#include <stdlib.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <iostream>

#define ITERATIONS 1000000

using namespace std;

struct ivec3 {
	int x;
	int y;
	int z;
	int w;
};

struct vec3 {
	float x;
	float y;
	float z;
	float w;
};

__m128i shufflePseudo(unsigned char (&a)[16], unsigned char (&b)[16]){
	unsigned char r[16];

	//INTEL
	for (int i = 0; i<16; i++){
		if (b[i] & 0x80)
			r[i] = 0;
		else
			r[i] = a[b[i] & 0x0F];
	}

	return _mm_loadu_si128((__m128i*)&r[0]);
}

__m128i shuffleInstruction(unsigned char (&a)[16], unsigned char(&b)[16]){
	__m128i ma = _mm_loadu_si128((__m128i*)&a[0]);
	__m128i mb = _mm_loadu_si128((__m128i*)&b[0]);
	return _mm_shuffle_epi8(ma, mb);
}

__m128i alignrPseudo(unsigned char (&a)[16], unsigned char (&b)[16], int n){
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
}

__m128i alignrInstruction(unsigned char (&a)[16], unsigned char (&b)[16], int num){
	__m128i ma = _mm_loadu_si128((__m128i*)&a[0]);
	__m128i mb = _mm_loadu_si128((__m128i*)&b[0]);

	__m128i r = _mm_alignr_epi8(ma, mb, 0);

	return r;
}


__m128i mulhrsPseudo( signed short (&a)[8], signed short (&b)[8]){
	signed short r[8];

	for (int i = 0; i < 8; i++) {
		r[i] =  (( (int)((a[i] * b[i]) >> 14) + 1) >> 1) & 0xFFFF;
	}

	return _mm_loadu_si128((__m128i*)&r[0]);
}


__m128i mulhrsInstruction(signed short (&a)[8], signed short (&b)[8]){
	__m128i ma = _mm_loadu_si128((__m128i*)&a[0]);
	__m128i mb = _mm_loadu_si128((__m128i*)&b[0]);
	return _mm_mulhrs_epi16(ma, mb);
}

int main (int argc, char* argv[]){

	__m128i mr1, mr2, mr3, mr4, mr5, mr6;
	StopWatch timer;

	unsigned char a[16];
	unsigned char b[16];

	for (int i = 0;i<16;i++){
		b[i] = (i<<1);
		a[i] = 128+i;
	}

	timer.start();
	for (int i = 0;i<ITERATIONS;i++){
		mr1 = shufflePseudo(a,b);
	}
	double elapsed = timer.stop();
	cout << "Elapsed time for Shuffle pseudo: " << elapsed << endl;

	timer.reset();
	timer.start();
	for (int i = 0;i<ITERATIONS;i++){
		mr2 = shuffleInstruction(a,b);
	}
	elapsed = timer.stop();
	cout << "Elapsed time for _mm_shuffle_epi8: " << elapsed << endl;

	//---------------

	if (_mm_test_all_ones(_mm_cmpeq_epi8(mr1, mr2)))
		cout<<"Values match"<<endl;

	timer.reset();
	timer.start();
	for (int i = 0;i<ITERATIONS;i++){
		mr3 = alignrPseudo(a,b,0);
	}
	elapsed = timer.stop();
	cout << "Elapsed time for alignrPseudo : " << elapsed << endl;

	timer.reset();
	timer.start();
	for (int i = 0;i<ITERATIONS;i++){
		mr4 = alignrInstruction(a,b,0);
	}
	elapsed = timer.stop();
	cout << "Elapsed time for _mm_alignr_epi8 : " << elapsed << endl;


	if (_mm_test_all_ones(_mm_cmpeq_epi8(mr3, mr4)))
		cout<<"Values match"<<endl;

	signed short ssa[8];
	signed short ssb[8];

	for (int i = 0;i<8;i++){
		ssa[i] = 1;
		ssb[i] = 2;
	}

	timer.reset();
	timer.start();
	for (int i = 0;i<ITERATIONS;i++){
		mr5 = mulhrsPseudo(ssa, ssb);
	}
	elapsed = timer.stop();
	cout << "Elapsed time for mulhrsPseudo : " << elapsed << endl;

	timer.reset();
	timer.start();
	for (int i = 0;i<ITERATIONS;i++){
		mr6 = mulhrsInstruction(ssa, ssb);	
	}
	elapsed = timer.stop();
	cout << "Elapsed time for _mm_mulhrs_epi16 : " << elapsed << endl;


	if (_mm_test_all_ones(_mm_cmpeq_epi8(mr5, mr6)))
		cout<<"Values match"<<endl;
	
	system("pause");
}