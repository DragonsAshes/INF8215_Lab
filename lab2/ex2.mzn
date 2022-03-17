% Retault Virgile : 2164296
% Foucher Sebastien : 2162248

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

constraint married[Juliette] < married[Charles];
constraint married[Juliette] < married[Tremblay];

constraint married[Juliette] > married[David];
constraint married[Juliette] > married[Dupont];

constraint married[Elisabeth] = 2 * married[Durand];
constraint 2 * married[Elisabeth] = married[Remy];

constraint married[Petit] = married[Pierre] + 10;
constraint married[Petit] = married[Maria] - 10;

constraint married[David] = married[Gagnon] - 25;
constraint married[David] = married[Manon];

constraint married[Gagnon] = 30;
constraint arg_max(married) = Gagnon;

constraint arg_min(married) != Emilie;
constraint arg_min(married) != Dupont;

constraint married[Edouard] = 25;

constraint alldifferent(married);
constraint alldifferent(F);
constraint alldifferent(M);

solve satisfy;

output [
  "M: ", show(M), "\n",
  "F: ", show(F), "\n",
  "married : ", show(married), "\n",
]
;

% output : M: [5, 3, 4, 1, 2]
%          F: [2, 5, 4, 1, 3]
%          married : [10, 25, 5, 30, 20]