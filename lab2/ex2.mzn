%-Enigme des mariages-

include "globals.mzn"; 

int: n = 5;

int: Dupont = 1;
int: Tremblay = 2;
int: Durand = 3;
int: Gagnon = 4;
int: Petit = 5;

var 1..n: Remy;
var 1..n: David;
var 1..n: Charles;
var 1..n: Pierre;
var 1..n: Edouard;
array[1..n] of var 1..n: M = [Remy, David, Charles, Pierre, Edouard];
 
var 1..n: Emilie;
var 1..n: Juliette;
var 1..n: Maria;
var 1..n: Elisabeth;
var 1..n: Manon;
array[1..n] of var 1..n: F = [Emilie, Juliette, Maria, Elisabeth, Manon];

array[1..n] of var 0..100: married;

% <A COMPLETER>

constraint married[F[Juliette]] < married[M[Charles]];
constraint married[F[Juliette]] < married[Tremblay];
constraint married[M[Charles]] != married[Tremblay];

constraint married[F[Juliette]] > married[M[David]];
constraint married[F[Juliette]] > married[Dupont];
constraint married[M[David]] != married[Dupont];

constraint married[F[Elisabeth]] = 2 * married[Durand];
constraint 2 * married[F[Elisabeth]] = married[M[Remy]];

constraint married[Petit] = married[M[Pierre]] + 10;
constraint married[Petit] = married[F[Maria]] - 10;

constraint married[M[David]] = married[Gagnon] - 25;
constraint married[M[David]] = married[F[Manon]];

constraint married[Gagnon] = 30;
constraint arg_max(married) = Gagnon;

constraint arg_min(married) != F[Emilie];
constraint arg_min(married) != Dupont;

constraint married[M[Edouard]] = 25;

constraint alldifferent(married);

solve satisfy;

output [
  "print arrays element per element :\n",
  "M: ", show(M[Remy]), ", ", show(M[David]), ", ", show(M[Charles]), ", ", show(M[Pierre]), ", ", show(M[Edouard]) , "\n",
  "F: ", show(F[Emilie]), ", ", show(F[Juliette]), ", ", show(F[Maria]), ", ", show(F[Elisabeth]), ", ", show(F[Manon]) , "\n",
  "print arrays with default print : \n",
  "M: ", show(M), "\n",
  "F: ", show(F), "\n",
  "married : ", show(married), "\n",
]
;