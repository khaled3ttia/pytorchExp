
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
br i1 %18, label %48, label %19
!i1B

	full_text


i1 %18
Ybitcast8BL
J
	full_text=
;
9%20 = bitcast double* %0 to [103 x [103 x [5 x double]]]*
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
	full_text?
?
~%27 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %20, i64 %22, i64 %24, i64 %26, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %20
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
?getelementptr8B?
?
	full_text?
?
~%29 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %20, i64 %22, i64 %24, i64 %26, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %20
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
Abitcast8B4
2
	full_text%
#
!%30 = bitcast double* %29 to i64*
-double*8B

	full_text

double* %29
Hload8B>
<
	full_text/
-
+%31 = load i64, i64* %30, align 8, !tbaa !8
'i64*8B

	full_text


i64* %30
?getelementptr8B?
?
	full_text?
?
~%32 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %20, i64 %22, i64 %24, i64 %26, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %20
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
1%33 = load double, double* %32, align 8, !tbaa !8
-double*8B

	full_text

double* %32
?getelementptr8B?
?
	full_text?
?
~%34 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %20, i64 %22, i64 %24, i64 %26, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %20
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
1%35 = load double, double* %34, align 8, !tbaa !8
-double*8B

	full_text

double* %34
?getelementptr8B?
?
	full_text?
?
~%36 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %20, i64 %22, i64 %24, i64 %26, i64 4
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %20
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
Ucall8BK
I
	full_text<
:
8%38 = tail call double @_Z4sqrtd(double 5.000000e-01) #2
7fmul8B-
+
	full_text

%39 = fmul double %28, %38
+double8B

	full_text


double %28
+double8B

	full_text


double %38
7fadd8B-
+
	full_text

%40 = fadd double %35, %37
+double8B

	full_text


double %35
+double8B

	full_text


double %37
@fmul8B6
4
	full_text'
%
#%41 = fmul double %40, 5.000000e-01
+double8B

	full_text


double %40
7fsub8B-
+
	full_text

%42 = fsub double %35, %37
+double8B

	full_text


double %35
+double8B

	full_text


double %37
7fmul8B-
+
	full_text

%43 = fmul double %38, %42
+double8B

	full_text


double %38
+double8B

	full_text


double %42
Nstore8BC
A
	full_text4
2
0store double %43, double* %27, align 8, !tbaa !8
+double8B

	full_text


double %43
-double*8B

	full_text

double* %27
Afsub8B7
5
	full_text(
&
$%44 = fsub double -0.000000e+00, %33
+double8B

	full_text


double %33
Nstore8BC
A
	full_text4
2
0store double %44, double* %29, align 8, !tbaa !8
+double8B

	full_text


double %44
-double*8B

	full_text

double* %29
Abitcast8B4
2
	full_text%
#
!%45 = bitcast double* %32 to i64*
-double*8B

	full_text

double* %32
Hstore8B=
;
	full_text.
,
*store i64 %31, i64* %45, align 8, !tbaa !8
%i648B

	full_text
	
i64 %31
'i64*8B

	full_text


i64* %45
7fsub8B-
+
	full_text

%46 = fsub double %41, %39
+double8B

	full_text


double %41
+double8B

	full_text


double %39
Nstore8BC
A
	full_text4
2
0store double %46, double* %34, align 8, !tbaa !8
+double8B

	full_text


double %46
-double*8B

	full_text

double* %34
7fadd8B-
+
	full_text

%47 = fadd double %39, %41
+double8B

	full_text


double %39
+double8B

	full_text


double %41
Nstore8BC
A
	full_text4
2
0store double %47, double* %36, align 8, !tbaa !8
+double8B

	full_text


double %47
-double*8B

	full_text

double* %36
'br8B

	full_text

br label %48
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %3
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
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 2
#i648B

	full_text	

i64 0
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 2
5double8B'
%
	full_text

double -0.000000e+00
4double8B&
$
	full_text

double 5.000000e-01
#i648B

	full_text	

i64 4        		 
 

                      !" !! #$ ## %& %% '( '' )* )) +, +- +. +/ ++ 01 00 23 24 25 26 22 78 77 9: 99 ;< ;= ;> ;? ;; @A @@ BC BD BE BF BB GH GG IJ IK IL IM II NO NN PP QR QS QQ TU TV TT WX WW YZ Y[ YY \] \^ \\ _` _a __ bc bb de df dd gh gg ij ik ii lm ln ll op oq oo rs rt rr uv uw uu xz { | }    	 
             " $# &
 (' * ,! -% .) /+ 1 3! 4% 5) 62 87 : <! =% >) ?; A C! D% E) FB H J! K% L) MI O0 RP SG UN VT XG ZN [P ]Y ^\ `+ a@ cb e2 f; h9 jg kW mQ nl pB qQ sW tr vI w y x y y  ~~	 ~~ 	P  P ~~  ~~ ? 	? 	? 	? 
	? 2? 	? +	? 	? !	? #	? %	? '	? )? 		? B	? ;? b? P	? W	? I"
pinvr"
_Z13get_global_idj"

_Z4sqrtd*?
npb-SP-pinvr.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?
 
transfer_bytes_log1p
|?A

wgsize
 

transfer_bytes	
????

devmap_label


wgsize_log1p
|?A