
#include "StopWatch.h"
#include "SSSE3Helper.h"
#include <stdlib.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <iostream>

#define ITERATIONS 1000000

using namespace std;

//forward declarations
__m128i employeeSchedule(bool RunOnHardware);
void shuffleTest();
void alignrTest();
void mulhrsTest();
void employeeTest();

__m128i mr1, mr2, mr3, mr4, mr5, mr6;
unsigned char a[16];
unsigned char b[16];
signed short ssa[8];
signed short ssb[8];

int main (int argc, char* argv[]){

	shuffleTest();
	cout<<endl;


	alignrTest();
	cout<<endl;

	mulhrsTest();
	cout<<endl;

	cout<<"--------------------"<<endl;

	employeeTest();

	system("pause");
}

void shuffleTest(){
	StopWatch timer;
	double elapsed;

	for (int i = 0;i<ITERATIONS;i++){
		for (int i = 0;i<16;i++){
			b[i] = rand()%255;
			a[i] = rand()%255;
		}
		timer.start();
		mr1 = SSSE3Helper::shufflePseudo(a,b);
		elapsed = timer.stop();
	}
	cout << "Elapsed time for shuffle pseudo: \t" << elapsed << endl;

	timer.reset();
	for (int i = 0;i<ITERATIONS;i++){
		for (int i = 0;i<16;i++){
			b[i] = rand()%255;
			a[i] = rand()%255;
		}
		timer.start();
		mr2 = SSSE3Helper::shuffleInstruction(a,b);
		elapsed = timer.stop();
	}
	cout << "Elapsed time for _mm_shuffle_epi8: \t" << elapsed << endl;
}

void alignrTest(){
	StopWatch timer;
	double elapsed;

	timer.reset();
	for (int i = 0;i<ITERATIONS;i++){
		for (int i = 0;i<16;i++){
			b[i] = rand()%255;
			a[i] = rand()%255;
		}
		timer.start();
		mr3 = SSSE3Helper::alignrPseudo(a,b,0);
		elapsed = timer.stop();
	}
	cout << "Elapsed time for alignr pseudo : \t" << elapsed << endl;

	timer.reset();
	for (int i = 0;i<ITERATIONS;i++){
		for (int i = 0;i<16;i++){
			b[i] = rand()%255;
			a[i] = rand()%255;
		}
		timer.start();
		mr4 = SSSE3Helper::alignrInstruction(a,b,0);
		elapsed = timer.stop();
	}
	cout << "Elapsed time for _mm_alignr_epi8 : \t" << elapsed << endl;
}

void mulhrsTest(){
	StopWatch timer;
	double elapsed;

	timer.reset();
	for (int i = 0;i<ITERATIONS;i++){
		for (int i = 0;i<8;i++){
			ssa[i] = rand()%255;
			ssb[i] = rand()%255;
		}
		timer.start();
		mr5 = SSSE3Helper::mulhrsPseudo(ssa, ssb);
		elapsed = timer.stop();
	}
	cout << "Elapsed time for mulhrs pseudo : \t" << elapsed << endl;

	timer.reset();
	for (int i = 0;i<ITERATIONS;i++){
		for (int i = 0;i<8;i++){
			ssa[i] = rand()%255;
			ssb[i] = rand()%255;
		}
		timer.start();
		mr6 = SSSE3Helper::mulhrsInstruction(ssa, ssb);
		elapsed = timer.stop();
	}
	cout << "Elapsed time for _mm_mulhrs_epi16 : \t" << elapsed << endl;
}

void employeeTest(){
	StopWatch timer;
	timer.start();
	for (int i = 0;i<ITERATIONS;i++){
		employeeSchedule(false);
	}
	double elapsed = timer.stop();
	cout<< "Employee Schedule Pseudo: \t\t"<< elapsed<<endl;

	timer.reset();
	timer.start();
	for (int i = 0;i<ITERATIONS;i++){
		employeeSchedule(true);
	}
	elapsed = timer.stop();
	cout<<"Employee Schedule Instruction: \t\t"<< elapsed<<endl;
}


__m128i employeeSchedule(bool RunOnHardware){
	unsigned char emp[16];//regular employees
	unsigned char man[16];//managers

	for (int i = 1;i<=16;i++){
		emp[i-1] = i;
		man[i-1] = i+0xA0;
	}

	unsigned char b[16];
	for (int i = 0;i<16;i++){
		b[i] = 0x80;
	}
	b[0] = rand()%16;
	b[1] = rand()%16;
	b[2] = rand()%16;
	__m128i manshuffle;
	if (RunOnHardware)
		manshuffle = SSSE3Helper::shuffleInstruction(man,b);
	else
		manshuffle = SSSE3Helper::shufflePseudo(man,b);

	//change the mask
	for (int i = 0;i<16;i++){
		b[i] = rand()%16;
	}
	b[0] = 0x80;
	b[1] = 0x80;
	b[2] = 0x80;
	__m128i empshuffle;
	if (RunOnHardware)
		empshuffle = SSSE3Helper::shuffleInstruction(emp,b);
	else
		empshuffle = SSSE3Helper::shufflePseudo(emp,b);

	return _mm_alignr_epi8(manshuffle, empshuffle, 3);

}
