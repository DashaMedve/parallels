Выбор типа: 
1. При использовании make вы можете выбрать, какой тип использовать, следующим образом:
  Находясь в директории, содержащей main.cpp, напишите:
       make float     -  если хотите, чтобы использовался float,
     или
       make double    -  если хотите, чтобы использовался double.

2. При использовании cmake вы можете выбрать, какой тип использовать, следующим образом:
  Находясь в директории, содержащей main.cpp, напишите:
       1. mkdir build
       2. cd build
       3. cmake -DTYPE=FLOAT ..      - если хотите, чтобы использовался float,
     или
       3. cmake -DTYPE=DOUBLE ..     - если хотите, чтобы использовался double,
       4. make


Результат:
  1. При выборе float, сумма равна -0.0277862
  2. При выборе double, сумма равна 4.55085e-11
   
