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