
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
br i1 %18, label %49, label %19
!i1B

	full_text


i1 %18
Ybitcast8BL
J
	full_text=
;
9%20 = bitcast double* %0 to [163 x [163 x [5 x double]]]*
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
	full_text{
y
w%27 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %20, i64 %22, i64 %24, i64 %26
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %20
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
pgetelementptr8B]
[
	full_textN
L
J%28 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
Gbitcast8B:
8
	full_text+
)
'%29 = bitcast [5 x double]* %27 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
Hload8B>
<
	full_text/
-
+%30 = load i64, i64* %29, align 8, !tbaa !8
'i64*8B

	full_text


i64* %29
?getelementptr8B?
?
	full_text?
?
~%31 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %20, i64 %22, i64 %24, i64 %26, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %20
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
1%32 = load double, double* %31, align 8, !tbaa !8
-double*8B

	full_text

double* %31
?getelementptr8B?
?
	full_text?
?
~%33 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %20, i64 %22, i64 %24, i64 %26, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %20
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
?getelementptr8B?
?
	full_text?
?
~%35 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %20, i64 %22, i64 %24, i64 %26, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %20
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
1%36 = load double, double* %35, align 8, !tbaa !8
-double*8B

	full_text

double* %35
?getelementptr8B?
?
	full_text?
?
~%37 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %20, i64 %22, i64 %24, i64 %26, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %20
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
1%38 = load double, double* %37, align 8, !tbaa !8
-double*8B

	full_text

double* %37
Ucall8BK
I
	full_text<
:
8%39 = tail call double @_Z4sqrtd(double 5.000000e-01) #2
7fmul8B-
+
	full_text

%40 = fmul double %34, %39
+double8B

	full_text


double %34
+double8B

	full_text


double %39
7fadd8B-
+
	full_text

%41 = fadd double %36, %38
+double8B

	full_text


double %36
+double8B

	full_text


double %38
@fmul8B6
4
	full_text'
%
#%42 = fmul double %41, 5.000000e-01
+double8B

	full_text


double %41
Afsub8B7
5
	full_text(
&
$%43 = fsub double -0.000000e+00, %32
+double8B

	full_text


double %32
Nstore8BC
A
	full_text4
2
0store double %43, double* %28, align 8, !tbaa !8
+double8B

	full_text


double %43
-double*8B

	full_text

double* %28
Abitcast8B4
2
	full_text%
#
!%44 = bitcast double* %31 to i64*
-double*8B

	full_text

double* %31
Hstore8B=
;
	full_text.
,
*store i64 %30, i64* %44, align 8, !tbaa !8
%i648B

	full_text
	
i64 %30
'i64*8B

	full_text


i64* %44
7fsub8B-
+
	full_text

%45 = fsub double %36, %38
+double8B

	full_text


double %36
+double8B

	full_text


double %38
7fmul8B-
+
	full_text

%46 = fmul double %39, %45
+double8B

	full_text


double %39
+double8B

	full_text


double %45
Nstore8BC
A
	full_text4
2
0store double %46, double* %33, align 8, !tbaa !8
+double8B

	full_text


double %46
-double*8B

	full_text

double* %33
7fsub8B-
+
	full_text

%47 = fsub double %42, %40
+double8B

	full_text


double %42
+double8B

	full_text


double %40
Nstore8BC
A
	full_text4
2
0store double %47, double* %35, align 8, !tbaa !8
+double8B

	full_text


double %47
-double*8B

	full_text

double* %35
7fadd8B-
+
	full_text

%48 = fadd double %40, %42
+double8B

	full_text


double %40
+double8B

	full_text


double %42
Nstore8BC
A
	full_text4
2
0store double %48, double* %37, align 8, !tbaa !8
+double8B

	full_text


double %48
-double*8B

	full_text

double* %37
'br8B

	full_text

br label %49
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %2
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
i32 %3
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 2
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 3
#i328B

	full_text	

i32 0
5double8B'
%
	full_text

double -0.000000e+00
#i648B

	full_text	

i64 2
#i648B

	full_text	

i64 4
#i648B

	full_text	

i64 0
4double8B&
$
	full_text

double 5.000000e-01        		 
 

                      !" !! #$ ## %& %% '( '' )* )) +, +- +. +/ ++ 01 00 23 22 45 44 67 68 69 6: 66 ;< ;; => =? =@ =A == BC BB DE DF DG DH DD IJ II KL KM KN KO KK PQ PP RR ST SU SS VW VX VV YZ YY [\ [[ ]^ ]_ ]] `a `` bc bd bb ef eg ee hi hj hh kl km kk no np nn qr qs qq tu tv tt wx wy ww z| } ~     	 
             " $# &
 (' * ,! -% .) /+ 1+ 32 5 7! 8% 9) :6 < >! ?% @) A= C E! F% G) HD J L! M% N) OK QB TR UI WP XV Z; \[ ^0 _6 a4 c` dI fP gR ie jh l= mY oS pn rD sS uY vt xK y { z { ?? ?? { ?? 	 ?? 	 ?? R ?? R	? 	? 	? 
	? 6? ? 	? 	? !	? #	? %	? '	? )	? D? 	? [	? =	? K	? 0	? 0? R	? Y"
ninvr"
_Z13get_global_idj"

_Z4sqrtd*?
npb-SP-ninvr.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

transfer_bytes	
????
 
transfer_bytes_log1p
??A

wgsize_log1p
??A

devmap_label


wgsize
@