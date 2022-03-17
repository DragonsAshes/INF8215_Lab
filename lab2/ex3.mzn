% Retault Virgile : 2164296
% Foucher Sebastien : 2162248

%-Attribution de taches-

include "globals.mzn";

array[1..N,1..S] of int: cost;%cout d'une tache dependamment du sous-traitant
array[1..N,1..S] of int: duration;%duree d'une tache dependamment du sous-traitant, en jours

int: S;%nombre de sous-traitants
int: N;%nombre de tâches
int: T;%horizon temporel, en jours

array[1..N] of var 1..S: x;

var int: C = sum([cost[i, x[i]] | i in 1..N]);

constraint forall(i in 2..N)( x[i] != x[1]);
constraint forall(i in 3..N)( x[i] != x[2]);

constraint max([sum([duration[i, s] | i in 1..N where x[i] = s]) | s in 1..S]) <= T;


solve minimize C;

% <A COMPLETER>

output [
  "x: ", show(x), "\n",
  "Cost: ", show(C)]
;

%output : x: [10, 7, 6, 3, 3, 2, 9, 4, 6, 5, 5, 8, 3, 5, 4, 1, 1, 9, 2, 8, 8, 6, 4, 1, 2]
%         Cost: 2304

% Execution avec la version 2.5.5 de MinizincIDE