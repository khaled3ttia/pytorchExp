
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
br i1 %21, label %22, label %63
!i1B

	full_text


i1 %21
Ybitcast8BL
J
	full_text=
;
9%23 = bitcast double* %0 to [103 x [103 x [5 x double]]]*
Sbitcast8BF
D
	full_text7
5
3%24 = bitcast double* %1 to [103 x [103 x double]]*
Sbitcast8BF
D
	full_text7
5
3%25 = bitcast double* %2 to [103 x [103 x double]]*
Sbitcast8BF
D
	full_text7
5
3%26 = bitcast double* %3 to [103 x [103 x double]]*
Sbitcast8BF
D
	full_text7
5
3%27 = bitcast double* %4 to [103 x [103 x double]]*
Sbitcast8BF
D
	full_text7
5
3%28 = bitcast double* %5 to [103 x [103 x double]]*
Sbitcast8BF
D
	full_text7
5
3%29 = bitcast double* %6 to [103 x [103 x double]]*
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
¨getelementptr8B”
‘
	full_textƒ
€
~%36 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %23, i64 %31, i64 %33, i64 %35, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %23
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
@fdiv8B6
4
	full_text'
%
#%38 = fdiv double 1.000000e+00, %37
+double8B

	full_text


double %37
‘getelementptr8B~
|
	full_texto
m
k%39 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %28, i64 %31, i64 %33, i64 %35
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %28
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
0store double %38, double* %39, align 8, !tbaa !8
+double8B

	full_text


double %38
-double*8B

	full_text

double* %39
¨getelementptr8B”
‘
	full_textƒ
€
~%40 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %23, i64 %31, i64 %33, i64 %35, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %23
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
7fmul8B-
+
	full_text

%42 = fmul double %38, %41
+double8B

	full_text


double %38
+double8B

	full_text


double %41
‘getelementptr8B~
|
	full_texto
m
k%43 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %24, i64 %31, i64 %33, i64 %35
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %24
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
0store double %42, double* %43, align 8, !tbaa !8
+double8B

	full_text


double %42
-double*8B

	full_text

double* %43
¨getelementptr8B”
‘
	full_textƒ
€
~%44 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %23, i64 %31, i64 %33, i64 %35, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %23
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
1%45 = load double, double* %44, align 8, !tbaa !8
-double*8B

	full_text

double* %44
7fmul8B-
+
	full_text

%46 = fmul double %38, %45
+double8B

	full_text


double %38
+double8B

	full_text


double %45
‘getelementptr8B~
|
	full_texto
m
k%47 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %25, i64 %31, i64 %33, i64 %35
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %25
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
¨getelementptr8B”
‘
	full_textƒ
€
~%48 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %23, i64 %31, i64 %33, i64 %35, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %23
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
1%49 = load double, double* %48, align 8, !tbaa !8
-double*8B

	full_text

double* %48
7fmul8B-
+
	full_text

%50 = fmul double %38, %49
+double8B

	full_text


double %38
+double8B

	full_text


double %49
‘getelementptr8B~
|
	full_texto
m
k%51 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %26, i64 %31, i64 %33, i64 %35
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %26
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
Nload8BD
B
	full_text5
3
1%52 = load double, double* %40, align 8, !tbaa !8
-double*8B

	full_text

double* %40
Nload8BD
B
	full_text5
3
1%53 = load double, double* %44, align 8, !tbaa !8
-double*8B

	full_text

double* %44
7fmul8B-
+
	full_text

%54 = fmul double %53, %53
+double8B

	full_text


double %53
+double8B

	full_text


double %53
icall8B_
]
	full_textP
N
L%55 = tail call double @llvm.fmuladd.f64(double %52, double %52, double %54)
+double8B

	full_text


double %52
+double8B

	full_text


double %52
+double8B

	full_text


double %54
Nload8BD
B
	full_text5
3
1%56 = load double, double* %48, align 8, !tbaa !8
-double*8B

	full_text

double* %48
icall8B_
]
	full_textP
N
L%57 = tail call double @llvm.fmuladd.f64(double %56, double %56, double %55)
+double8B

	full_text


double %56
+double8B

	full_text


double %56
+double8B

	full_text


double %55
@fmul8B6
4
	full_text'
%
#%58 = fmul double %57, 5.000000e-01
+double8B

	full_text


double %57
7fmul8B-
+
	full_text

%59 = fmul double %38, %58
+double8B

	full_text


double %38
+double8B

	full_text


double %58
‘getelementptr8B~
|
	full_texto
m
k%60 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %29, i64 %31, i64 %33, i64 %35
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %29
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
0store double %59, double* %60, align 8, !tbaa !8
+double8B

	full_text


double %59
-double*8B

	full_text

double* %60
7fmul8B-
+
	full_text

%61 = fmul double %38, %59
+double8B

	full_text


double %38
+double8B

	full_text


double %59
‘getelementptr8B~
|
	full_texto
m
k%62 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %27, i64 %31, i64 %33, i64 %35
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %27
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
0store double %61, double* %62, align 8, !tbaa !8
+double8B

	full_text


double %61
-double*8B

	full_text

double* %62
'br8B

	full_text

br label %63
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %9
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %2
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
$i328B

	full_text


i32 %8
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %3
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
#i328B

	full_text	

i32 0
4double8B&
$
	full_text

double 1.000000e+00
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
4double8B&
$
	full_text

double 5.000000e-01
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
#i648B

	full_text	

i64 1       	  
 

                      !" !! #$ ## %& %% '( '' )* )) +, +- +. +/ ++ 01 00 23 22 45 46 47 48 44 9: 9; 99 <= <> <? <@ << AB AA CD CE CC FG FH FI FJ FF KL KM KK NO NP NQ NR NN ST SS UV UW UU XY XZ X[ X\ XX ]^ ]_ ]] `a `b `c `d `` ef ee gh gi gg jk jl jm jn jj op oq oo rs rr tu tt vw vx vv yz y{ y| yy }~ }} € 	 	‚  ƒ„ ƒƒ …† …
‡ …… ˆ‰ ˆ
Š ˆ
‹ ˆ
Œ ˆˆ Ž 
  ‘ 
’  “” “
• “
– “
— ““ ˜™ ˜
š ˜˜ ›	 ž Ÿ   ¡ ¢ 	£ ¤ ¥ 	¦    	 
          " $# & (' * ,! -% .) /+ 10 3 5! 6% 7) 82 :4 ; =! >% ?) @< B2 DA E G! H% I) JC LF M O! P% Q) RN T2 VS W Y! Z% [) \U ^X _ a! b% c) d` f2 he i k! l% m) ng pj q< sN ut wt xr zr {v |` ~} €} y ‚ „2 †ƒ ‡ ‰! Š% ‹) Œ… Žˆ 2 ‘… ’ ”! •% –) — ™“ š  œ› œ §§ ¨¨ œy ¨¨ y §§  §§  §§  ¨¨ © ª 2	« `	¬ N
­ ƒ® ¯ 	° +	± 	± !	± #	± %	± '	± )	² <"
compute_rhs1"
_Z13get_global_idj"
llvm.fmuladd.f64*•
npb-3.3-BT-compute_rhs1_B.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282

wgsize
 

wgsize_log1p
 ø§A

transfer_bytes	
˜®´ò
 
transfer_bytes_log1p
 ø§A

devmap_label
