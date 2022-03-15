

[external]
LcallBD
B
	full_text5
3
1%11 = tail call i64 @_Z13get_global_idj(i32 2) #3
6truncB-
+
	full_text

%12 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
LcallBD
B
	full_text5
3
1%13 = tail call i64 @_Z13get_global_idj(i32 1) #3
LcallBD
B
	full_text5
3
1%14 = tail call i64 @_Z13get_global_idj(i32 0) #3
6truncB-
+
	full_text

%15 = trunc i64 %14 to i32
#i64B

	full_text
	
i64 %14
5icmpB-
+
	full_text

%16 = icmp slt i32 %12, %9
#i32B

	full_text
	
i32 %12
6truncB-
+
	full_text

%17 = trunc i64 %13 to i32
#i64B

	full_text
	
i64 %13
5icmpB-
+
	full_text

%18 = icmp slt i32 %17, %8
#i32B

	full_text
	
i32 %17
/andB(
&
	full_text

%19 = and i1 %16, %18
!i1B

	full_text


i1 %16
!i1B

	full_text


i1 %18
5icmpB-
+
	full_text

%20 = icmp slt i32 %15, %7
#i32B

	full_text
	
i32 %15
/andB(
&
	full_text

%21 = and i1 %19, %20
!i1B

	full_text


i1 %19
!i1B

	full_text


i1 %20
8brB2
0
	full_text#
!
br i1 %21, label %22, label %60
!i1B

	full_text


i1 %21
Wbitcast8BJ
H
	full_text;
9
7%23 = bitcast double* %0 to [13 x [13 x [5 x double]]]*
Qbitcast8BD
B
	full_text5
3
1%24 = bitcast double* %1 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%25 = bitcast double* %2 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%26 = bitcast double* %3 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%27 = bitcast double* %4 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%28 = bitcast double* %5 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%29 = bitcast double* %6 to [13 x [13 x double]]*
1shl8B(
&
	full_text

%30 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%31 = ashr exact i64 %30, 32
%i648B

	full_text
	
i64 %30
1shl8B(
&
	full_text

%32 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
9ashr8B/
-
	full_text 

%33 = ashr exact i64 %32, 32
%i648B

	full_text
	
i64 %32
1shl8B(
&
	full_text

%34 = shl i64 %14, 32
%i648B

	full_text
	
i64 %14
9ashr8B/
-
	full_text 

%35 = ashr exact i64 %34, 32
%i648B

	full_text
	
i64 %34
¢getelementptr8Bé
ã
	full_text~
|
z%36 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %23, i64 %31, i64 %33, i64 %35, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %23
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nload8BD
B
	full_text5
3
1%37 = load double, double* %36, align 8, !tbaa !8
-double*8B

	full_text

double* %36
¢getelementptr8Bé
ã
	full_text~
|
z%38 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %23, i64 %31, i64 %33, i64 %35, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %23
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nload8BD
B
	full_text5
3
1%39 = load double, double* %38, align 8, !tbaa !8
-double*8B

	full_text

double* %38
¢getelementptr8Bé
ã
	full_text~
|
z%40 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %23, i64 %31, i64 %33, i64 %35, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %23
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nload8BD
B
	full_text5
3
1%41 = load double, double* %40, align 8, !tbaa !8
-double*8B

	full_text

double* %40
¢getelementptr8Bé
ã
	full_text~
|
z%42 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %23, i64 %31, i64 %33, i64 %35, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %23
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nload8BD
B
	full_text5
3
1%43 = load double, double* %42, align 8, !tbaa !8
-double*8B

	full_text

double* %42
@fdiv8B6
4
	full_text'
%
#%44 = fdiv double 1.000000e+00, %37
+double8B

	full_text


double %37
çgetelementptr8Bz
x
	full_textk
i
g%45 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %28, i64 %31, i64 %33, i64 %35
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %28
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nstore8BC
A
	full_text4
2
0store double %44, double* %45, align 8, !tbaa !8
+double8B

	full_text


double %44
-double*8B

	full_text

double* %45
7fmul8B-
+
	full_text

%46 = fmul double %39, %44
+double8B

	full_text


double %39
+double8B

	full_text


double %44
çgetelementptr8Bz
x
	full_textk
i
g%47 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %24, i64 %31, i64 %33, i64 %35
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %24
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nstore8BC
A
	full_text4
2
0store double %46, double* %47, align 8, !tbaa !8
+double8B

	full_text


double %46
-double*8B

	full_text

double* %47
7fmul8B-
+
	full_text

%48 = fmul double %44, %41
+double8B

	full_text


double %44
+double8B

	full_text


double %41
çgetelementptr8Bz
x
	full_textk
i
g%49 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %25, i64 %31, i64 %33, i64 %35
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %25
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nstore8BC
A
	full_text4
2
0store double %48, double* %49, align 8, !tbaa !8
+double8B

	full_text


double %48
-double*8B

	full_text

double* %49
7fmul8B-
+
	full_text

%50 = fmul double %44, %43
+double8B

	full_text


double %44
+double8B

	full_text


double %43
çgetelementptr8Bz
x
	full_textk
i
g%51 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %26, i64 %31, i64 %33, i64 %35
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %26
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nstore8BC
A
	full_text4
2
0store double %50, double* %51, align 8, !tbaa !8
+double8B

	full_text


double %50
-double*8B

	full_text

double* %51
7fmul8B-
+
	full_text

%52 = fmul double %41, %41
+double8B

	full_text


double %41
+double8B

	full_text


double %41
icall8B_
]
	full_textP
N
L%53 = tail call double @llvm.fmuladd.f64(double %39, double %39, double %52)
+double8B

	full_text


double %39
+double8B

	full_text


double %39
+double8B

	full_text


double %52
icall8B_
]
	full_textP
N
L%54 = tail call double @llvm.fmuladd.f64(double %43, double %43, double %53)
+double8B

	full_text


double %43
+double8B

	full_text


double %43
+double8B

	full_text


double %53
@fmul8B6
4
	full_text'
%
#%55 = fmul double %54, 5.000000e-01
+double8B

	full_text


double %54
7fmul8B-
+
	full_text

%56 = fmul double %44, %55
+double8B

	full_text


double %44
+double8B

	full_text


double %55
çgetelementptr8Bz
x
	full_textk
i
g%57 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %29, i64 %31, i64 %33, i64 %35
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %29
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nstore8BC
A
	full_text4
2
0store double %56, double* %57, align 8, !tbaa !8
+double8B

	full_text


double %56
-double*8B

	full_text

double* %57
7fmul8B-
+
	full_text

%58 = fmul double %44, %56
+double8B

	full_text


double %44
+double8B

	full_text


double %56
çgetelementptr8Bz
x
	full_textk
i
g%59 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %27, i64 %31, i64 %33, i64 %35
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %27
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
Nstore8BC
A
	full_text4
2
0store double %58, double* %59, align 8, !tbaa !8
+double8B

	full_text


double %58
-double*8B

	full_text

double* %59
'br8B

	full_text

br label %60
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %3
,double*8B

	full_text


double* %5
,double*8B

	full_text


double* %6
,double*8B

	full_text


double* %4
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %8
,double*8B

	full_text


double* %2
$i328B

	full_text


i32 %9
$i328B

	full_text


i32 %7
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
$i648B

	full_text


i64 32
4double8B&
$
	full_text

double 5.000000e-01
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
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 2
4double8B&
$
	full_text

double 1.000000e+00       	  
 

                      !" !! #$ ## %& %% '( '' )* )) +, +- +. +/ ++ 01 00 23 24 25 26 22 78 77 9: 9; 9< 9= 99 >? >> @A @B @C @D @@ EF EE GH GG IJ IK IL IM II NO NP NN QR QS QQ TU TV TW TX TT YZ Y[ YY \] \^ \\ _` _a _b _c __ de df dd gh gi gg jk jl jm jn jj op oq oo rs rt rr uv uw ux uu yz y{ y| yy }~ }} Ä 	Å  ÇÉ Ç
Ñ Ç
Ö Ç
Ü ÇÇ áà á
â áá äã ä
å ää çé ç
è ç
ê ç
ë çç íì í
î íí ïó ò ô ö õ ú 	ù û 	ü 	†    	 
          " $# & (' * ,! -% .) /+ 1 3! 4% 5) 62 8 :! ;% <) =9 ? A! B% C) D@ F0 H J! K% L) MG OI P7 RG S U! V% W) XQ ZT [G ]> ^ `! a% b) c\ e_ fG hE i k! l% m) ng pj q> s> t7 v7 wr xE zE {u |y ~G Ä} Å É! Ñ% Ö) Ü àÇ âG ã å é! è% ê) ëä ìç î  ñï ñ ¢¢ ñ °°y ¢¢ y °°  °°  °° u ¢¢ u	£ 	£ !	£ #	£ %	£ '	£ )	§ }	• +	¶ @	ß 2® © ™ 	´ 9¨ G"
compute_rhs1"
_Z13get_global_idj"
llvm.fmuladd.f64*ë
npb-BT-compute_rhs1_S.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282

wgsize
<

wgsize_log1p
ÜfA

devmap_label
 

transfer_bytes
¯¨n
 
transfer_bytes_log1p
ÜfA