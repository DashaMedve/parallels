parallel_test: task3.o
	g++-13 task3.o -D VAR1 -fopenmp -o parallel_test

task3.o: task3.cpp
	g++-13 -c -fopenmp task3.cpp

clean:
	rm task3.o parallel_test
