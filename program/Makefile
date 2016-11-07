CC=nvcc
FLAGS=-lcurand -rdc=true -use_fast_math -O3 
FOLDR=src

mish.out: constants.o config.o mish.o main.o
	$(CC) main.o mish.o config.o constants.o -o mish.out

main.o: $(FOLDR)/main.cu 
	$(CC) $(FOLDR)/main.cu -c $(FLAGS)

mish.o: $(FOLDR)/mish.cu $(FOLDR)/kernels.cuh $(FOLDR)/mish.cuh
	$(CC) $(FOLDR)/mish.cu -c $(FLAGS)

config.o: $(FOLDR)/config.cu $(FOLDR)/config.cuh
	$(CC) $(FOLDR)/config.cu -c $(FLAGS)

constants.o: $(FOLDR)/constants.cu $(FOLDR)/constants.cuh
	$(CC) $(FOLDR)/constants.cu -c $(FLAGS)

clean:
	rm *.o
	rm *.out

