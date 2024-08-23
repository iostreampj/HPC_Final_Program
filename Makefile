all: program2

program2:program2.cpp
	mpic++ -std=c++11  -O2  program2.cpp  -o program2
clean:
	rm -f program2 *.o
	
