

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 2) #2
,addB%
#
	full_text

%6 = add i64 %5, 1
"i64B

	full_text


i64 %5
4truncB+
)
	full_text

%7 = trunc i64 %6 to i32
"i64B

	full_text


i64 %6
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 1) #2
,addB%
#
	full_text

%9 = add i64 %8, 1
"i64B

	full_text


i64 %8
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 0) #2
.addB'
%
	full_text

%11 = add i64 %10, 1
#i64B

	full_text
	
i64 %10
6truncB-
+
	full_text

%12 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
4icmpB,
*
	full_text

%13 = icmp sgt i32 %7, %3
"i32B

	full_text


i32 %7
5truncB,
*
	full_text

%14 = trunc i64 %9 to i32
"i64B

	full_text


i64 %9
5icmpB-
+
	full_text

%15 = icmp sgt i32 %14, %2
#i32B

	full_text
	
i32 %14
-orB'
%
	full_text

%16 = or i1 %13, %15
!i1B

	full_text


i1 %13
!i1B

	full_text


i1 %15
5icmpB-
+
	full_text

%17 = icmp sgt i32 %12, %1
#i32B

	full_text
	
i32 %12
-orB'
%
	full_text

%18 = or i1 %16, %17
!i1B

	full_text


i1 %16
!i1B

	full_text


i1 %17
8brB2
0
	full_text#
!
br i1 %18, label %42, label %19
!i1B

	full_text


i1 %18
Wbitcast8BJ
H
	full_text;
9
7%20 = bitcast double* %0 to [65 x [65 x [5 x double]]]*
0shl8B'
%
	full_text

%21 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%22 = ashr exact i64 %21, 32
%i648B

	full_text
	
i64 %21
0shl8B'
%
	full_text

%23 = shl i64 %9, 32
$i648B

	full_text


i64 %9
9ashr8B/
-
	full_text 

%24 = ashr exact i64 %23, 32
%i648B

	full_text
	
i64 %23
1shl8B(
&
	full_text

%25 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%26 = ashr exact i64 %25, 32
%i648B

	full_text
	
i64 %25
?getelementptr8B?
?
	full_text~
|
z%27 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %22, i64 %24, i64 %26, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
Nload8BD
B
	full_text5
3
1%28 = load double, double* %27, align 8, !tbaa !8
-double*8B

	full_text

double* %27
@fmul8B6
4
	full_text'
%
#%29 = fmul double %28, 1.500000e-03
+double8B

	full_text


double %28
Nstore8BC
A
	full_text4
2
0store double %29, double* %27, align 8, !tbaa !8
+double8B

	full_text


double %29
-double*8B

	full_text

double* %27
?getelementptr8B?
?
	full_text~
|
z%30 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %22, i64 %24, i64 %26, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
Nload8BD
B
	full_text5
3
1%31 = load double, double* %30, align 8, !tbaa !8
-double*8B

	full_text

double* %30
@fmul8B6
4
	full_text'
%
#%32 = fmul double %31, 1.500000e-03
+double8B

	full_text


double %31
Nstore8BC
A
	full_text4
2
0store double %32, double* %30, align 8, !tbaa !8
+double8B

	full_text


double %32
-double*8B

	full_text

double* %30
?getelementptr8B?
?
	full_text~
|
z%33 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %22, i64 %24, i64 %26, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
Nload8BD
B
	full_text5
3
1%34 = load double, double* %33, align 8, !tbaa !8
-double*8B

	full_text

double* %33
@fmul8B6
4
	full_text'
%
#%35 = fmul double %34, 1.500000e-03
+double8B

	full_text


double %34
Nstore8BC
A
	full_text4
2
0store double %35, double* %33, align 8, !tbaa !8
+double8B

	full_text


double %35
-double*8B

	full_text

double* %33
?getelementptr8B?
?
	full_text~
|
z%36 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %22, i64 %24, i64 %26, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
Nload8BD
B
	full_text5
3
1%37 = load double, double* %36, align 8, !tbaa !8
-double*8B

	full_text

double* %36
@fmul8B6
4
	full_text'
%
#%38 = fmul double %37, 1.500000e-03
+double8B

	full_text


double %37
Nstore8BC
A
	full_text4
2
0store double %38, double* %36, align 8, !tbaa !8
+double8B

	full_text


double %38
-double*8B

	full_text

double* %36
?getelementptr8B?
?
	full_text~
|
z%39 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %22, i64 %24, i64 %26, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %26
Nload8BD
B
	full_text5
3
1%40 = load double, double* %39, align 8, !tbaa !8
-double*8B

	full_text

double* %39
@fmul8B6
4
	full_text'
%
#%41 = fmul double %40, 1.500000e-03
+double8B

	full_text


double %40
Nstore8BC
A
	full_text4
2
0store double %41, double* %39, align 8, !tbaa !8
+double8B

	full_text


double %41
-double*8B

	full_text

double* %39
'br8B

	full_text

br label %42
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %1
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %3
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 3
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 2
4double8B&
$
	full_text

double 1.500000e-03
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 4        		 
 

                      !" !! #$ ## %& %% '( '' )* )) +, +- +. +/ ++ 01 00 23 22 45 46 44 78 79 7: 7; 77 <= << >? >> @A @B @@ CD CE CF CG CC HI HH JK JJ LM LN LL OP OQ OR OS OO TU TT VW VV XY XZ XX [\ [] [^ [_ [[ `a `` bc bb de df dd gi j k l    	 
             " $# &
 (' * ,! -% .) /+ 10 32 5+ 6 8! 9% :) ;7 =< ?> A7 B D! E% F) GC IH KJ MC N P! Q% R) SO UT WV YO Z \! ]% ^) _[ a` cb e[ f h g h mm h mm  mm 	 mm 	n 	o o o 
o 7p +q Or s Ct u 2u >u Ju Vu bv v !v #v %v 'v )w ["
compute_rhs6"
_Z13get_global_idj*?
npb-SP-compute_rhs6.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?
 
transfer_bytes_log1p
?Y?A

transfer_bytes
???5

devmap_label
 

wgsize
>

wgsize_log1p
?Y?A