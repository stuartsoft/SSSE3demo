
#include "StopWatch.h"
#include "SSSE3Helper.h"
#include <stdlib.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <iostream>

#define ITERATIONS 1000000

using namespace std;

//forward declarations

//EmployeeSchedule(bool) is used to generate a company schedule, represented by a 16 byte array in the form of a __m128i type.
//This function generates an array of 16 employee ids and 16 manager ids. It then generates a random 16 byte
//schedule mask (b) to dictate the schema of the schedule for Employees and Managers. The schedule mask is
//be used to generate shedules schemes following a pattern, as well as omit a certain number of employees from the resulting list.
//So 1 particular mask could be used for making a regular company schedule, and another mask could be used for 
//a holiday schedule. This would allow the schedule template to operate independently from the employee list
//and make it very easy to update the schedule if the employee list changed, while still retaining the 
//schedule scheme. Once the Employee schedule and manager shcedules are built, they are merged together using
//the alignr instruction and returned by the function as a 16 byte array represented as a __m128i.
__m128i employeeSchedule(bool RunOnHardware);

//a test built to compare software vs hardware implementations of shuffling a byte array based on another byte array mask
void shuffleTest();

//a test built to compare software vs hardware implementations of concatinating, byte shifting, and trunkating back to a 128 bit output
void alignrTest();

//a test built to compare software vs hardware implementations of the _m_mulhrs_epi8 operation
void mulhrsTest();

//a test built to demonstrate a real world example of using the shuffle and alignr operations in tandem and 
//the dramatic time savings when using the hardware implementation
void employeeTest();

__m128i mr1, mr2, mr3, mr4, mr5, mr6;
unsigned char a[16];
unsigned char b[16];
signed short ssa[8];
signed short ssb[8];

int main (int argc, char* argv[]){

	//Run shuffle calculation, comparing the runtime results for the software implementation and hardware instruction
	shuffleTest();
	cout<<endl;

	//run alignr calculation, again comparing results
	alignrTest();
	cout<<endl;

	//run mulhrs calculation, again comparing results
	mulhrsTest();
	cout<<endl;

	cout<<"--------------------"<<endl;

	//run the employee scheduler example, again comparing results
	employeeTest();

	system("pause");
}

void shuffleTest(){
	StopWatch timer;
	double elapsed;
	//software implementation
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

	//hardware implementation
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

	//software implementation
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

	//hardware implementation
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

	//software implementation
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

	//hardware implementation
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
	//software implementation
	timer.start();
	for (int i = 0;i<ITERATIONS;i++){
		employeeSchedule(false);
	}
	double elapsed = timer.stop();
	cout<< "Employee Schedule Pseudo: \t\t"<< elapsed<<endl;

	//hardware implementation
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

	__m128i r =  _mm_alignr_epi8(manshuffle, empshuffle, 3);
	//break point right here to view the memory layout and results of the employee schedule
	return r;

}
