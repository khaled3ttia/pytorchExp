

[external]
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 2) #3
5truncB,
*
	full_text

%10 = trunc i64 %9 to i32
"i64B

	full_text


i64 %9
LcallBD
B
	full_text5
3
1%11 = tail call i64 @_Z13get_global_idj(i32 1) #3
LcallBD
B
	full_text5
3
1%12 = tail call i64 @_Z13get_global_idj(i32 0) #3
6truncB-
+
	full_text

%13 = trunc i64 %12 to i32
#i64B

	full_text
	
i64 %12
5icmpB-
+
	full_text

%14 = icmp slt i32 %10, %7
#i32B

	full_text
	
i32 %10
6truncB-
+
	full_text

%15 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
5icmpB-
+
	full_text

%16 = icmp slt i32 %15, %6
#i32B

	full_text
	
i32 %15
/andB(
&
	full_text

%17 = and i1 %14, %16
!i1B

	full_text


i1 %14
!i1B

	full_text


i1 %16
5icmpB-
+
	full_text

%18 = icmp slt i32 %13, %5
#i32B

	full_text
	
i32 %13
/andB(
&
	full_text

%19 = and i1 %17, %18
!i1B

	full_text


i1 %17
!i1B

	full_text


i1 %18
8brB2
0
	full_text#
!
br i1 %19, label %20, label %68
!i1B

	full_text


i1 %19
Ybitcast8BL
J
	full_text=
;
9%21 = bitcast double* %1 to [103 x [103 x [5 x double]]]*
Ybitcast8BL
J
	full_text=
;
9%22 = bitcast double* %2 to [103 x [103 x [5 x double]]]*
Sbitcast8BF
D
	full_text7
5
3%23 = bitcast double* %3 to [103 x [103 x double]]*
0shl8B'
%
	full_text

%24 = shl i64 %9, 32
$i648B

	full_text


i64 %9
9ashr8B/
-
	full_text 

%25 = ashr exact i64 %24, 32
%i648B

	full_text
	
i64 %24
1shl8B(
&
	full_text

%26 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%27 = ashr exact i64 %26, 32
%i648B

	full_text
	
i64 %26
1shl8B(
&
	full_text

%28 = shl i64 %12, 32
%i648B

	full_text
	
i64 %12
9ashr8B/
-
	full_text 

%29 = ashr exact i64 %28, 32
%i648B

	full_text
	
i64 %28
®getelementptr8Bî
ë
	full_textÉ
Ä
~%30 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %22, i64 %25, i64 %27, i64 %29, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%31 = load double, double* %30, align 8, !tbaa !8
-double*8B

	full_text

double* %30
Afsub8B7
5
	full_text(
&
$%32 = fsub double -0.000000e+00, %31
+double8B

	full_text


double %31
®getelementptr8Bî
ë
	full_textÉ
Ä
~%33 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %21, i64 %25, i64 %27, i64 %29, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nstore8BC
A
	full_text4
2
0store double %32, double* %33, align 8, !tbaa !8
+double8B

	full_text


double %32
-double*8B

	full_text

double* %33
®getelementptr8Bî
ë
	full_textÉ
Ä
~%34 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %22, i64 %25, i64 %27, i64 %29, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%35 = load double, double* %34, align 8, !tbaa !8
-double*8B

	full_text

double* %34
Afsub8B7
5
	full_text(
&
$%36 = fsub double -0.000000e+00, %35
+double8B

	full_text


double %35
®getelementptr8Bî
ë
	full_textÉ
Ä
~%37 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %21, i64 %25, i64 %27, i64 %29, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nstore8BC
A
	full_text4
2
0store double %36, double* %37, align 8, !tbaa !8
+double8B

	full_text


double %36
-double*8B

	full_text

double* %37
®getelementptr8Bî
ë
	full_textÉ
Ä
~%38 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %22, i64 %25, i64 %27, i64 %29, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%39 = load double, double* %38, align 8, !tbaa !8
-double*8B

	full_text

double* %38
Afsub8B7
5
	full_text(
&
$%40 = fsub double -0.000000e+00, %39
+double8B

	full_text


double %39
®getelementptr8Bî
ë
	full_textÉ
Ä
~%41 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %21, i64 %25, i64 %27, i64 %29, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nstore8BC
A
	full_text4
2
0store double %40, double* %41, align 8, !tbaa !8
+double8B

	full_text


double %40
-double*8B

	full_text

double* %41
®getelementptr8Bî
ë
	full_textÉ
Ä
~%42 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %22, i64 %25, i64 %27, i64 %29, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%43 = load double, double* %42, align 8, !tbaa !8
-double*8B

	full_text

double* %42
Afsub8B7
5
	full_text(
&
$%44 = fsub double -0.000000e+00, %43
+double8B

	full_text


double %43
®getelementptr8Bî
ë
	full_textÉ
Ä
~%45 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %21, i64 %25, i64 %27, i64 %29, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
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
®getelementptr8Bî
ë
	full_textÉ
Ä
~%46 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %22, i64 %25, i64 %27, i64 %29, i64 4
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%47 = load double, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
Afsub8B7
5
	full_text(
&
$%48 = fsub double -0.000000e+00, %47
+double8B

	full_text


double %47
®getelementptr8Bî
ë
	full_textÉ
Ä
~%49 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %21, i64 %25, i64 %27, i64 %29, i64 4
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
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
Ybitcast8BL
J
	full_text=
;
9%50 = bitcast double* %0 to [103 x [103 x [5 x double]]]*
Sbitcast8BF
D
	full_text7
5
3%51 = bitcast double* %4 to [103 x [103 x double]]*
®getelementptr8Bî
ë
	full_textÉ
Ä
~%52 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %50, i64 %25, i64 %27, i64 %29, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %50
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%53 = load double, double* %52, align 8, !tbaa !8
-double*8B

	full_text

double* %52
@fdiv8B6
4
	full_text'
%
#%54 = fdiv double 1.000000e+00, %53
+double8B

	full_text


double %53
ëgetelementptr8B~
|
	full_texto
m
k%55 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %51, i64 %25, i64 %27, i64 %29
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %51
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nstore8BC
A
	full_text4
2
0store double %54, double* %55, align 8, !tbaa !8
+double8B

	full_text


double %54
-double*8B

	full_text

double* %55
®getelementptr8Bî
ë
	full_textÉ
Ä
~%56 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %50, i64 %25, i64 %27, i64 %29, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %50
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%57 = load double, double* %56, align 8, !tbaa !8
-double*8B

	full_text

double* %56
®getelementptr8Bî
ë
	full_textÉ
Ä
~%58 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %50, i64 %25, i64 %27, i64 %29, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %50
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%59 = load double, double* %58, align 8, !tbaa !8
-double*8B

	full_text

double* %58
7fmul8B-
+
	full_text

%60 = fmul double %59, %59
+double8B

	full_text


double %59
+double8B

	full_text


double %59
icall8B_
]
	full_textP
N
L%61 = tail call double @llvm.fmuladd.f64(double %57, double %57, double %60)
+double8B

	full_text


double %57
+double8B

	full_text


double %57
+double8B

	full_text


double %60
®getelementptr8Bî
ë
	full_textÉ
Ä
~%62 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %50, i64 %25, i64 %27, i64 %29, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %50
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%63 = load double, double* %62, align 8, !tbaa !8
-double*8B

	full_text

double* %62
icall8B_
]
	full_textP
N
L%64 = tail call double @llvm.fmuladd.f64(double %63, double %63, double %61)
+double8B

	full_text


double %63
+double8B

	full_text


double %63
+double8B

	full_text


double %61
@fmul8B6
4
	full_text'
%
#%65 = fmul double %64, 5.000000e-01
+double8B

	full_text


double %64
7fmul8B-
+
	full_text

%66 = fmul double %54, %65
+double8B

	full_text


double %54
+double8B

	full_text


double %65
ëgetelementptr8B~
|
	full_texto
m
k%67 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %23, i64 %25, i64 %27, i64 %29
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %23
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nstore8BC
A
	full_text4
2
0store double %66, double* %67, align 8, !tbaa !8
+double8B

	full_text


double %66
-double*8B

	full_text

double* %67
'br8B

	full_text

br label %68
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %2
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %3
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %5
,double*8B

	full_text


double* %4
$i328B

	full_text


i32 %7
$i328B

	full_text


i32 %6
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
5double8B'
%
	full_text

double -0.000000e+00
#i648B

	full_text	

i64 4
4double8B&
$
	full_text

double 1.000000e+00
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 3
4double8B&
$
	full_text

double 5.000000e-01
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
i64 2
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 2       	  
 

                      !" !! #$ ## %& %% '( ') '* '+ '' ,- ,, ./ .. 01 02 03 04 00 56 57 55 89 8: 8; 8< 88 => == ?@ ?? AB AC AD AE AA FG FH FF IJ IK IL IM II NO NN PQ PP RS RT RU RV RR WX WY WW Z[ Z\ Z] Z^ ZZ _` __ ab aa cd ce cf cg cc hi hj hh kl km kn ko kk pq pp rs rr tu tv tw tx tt yz y{ yy || }} ~ ~	Ä ~	Å ~	Ç ~~ ÉÑ ÉÉ Ö
Ü ÖÖ áà á
â á
ä á
ã áá åç å
é åå èê è
ë è
í è
ì èè îï îî ñó ñ
ò ñ
ô ñ
ö ññ õú õõ ùû ù
ü ùù †° †
¢ †
£ †† §• §
¶ §
ß §
® §§ ©™ ©© ´¨ ´
≠ ´
Æ ´´ Ø∞ ØØ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥
∑ ¥
∏ ¥¥ π∫ π
ª ππ ºæ ø ¿ ¡ |	¬ √ }	ƒ 	≈    	 
            " $# & ( )! *% +' -, / 1 2! 3% 4. 60 7 9 :! ;% <8 >= @ B C! D% E? GA H J K! L% MI ON Q S T! U% VP XR Y [ \! ]% ^Z `_ b d e! f% ga ic j l m! n% ok qp s u v! w% xr zt {|  Ä! Å% Ç~ ÑÉ Ü} à â! ä% ãÖ çá é| ê ë! í% ìè ï| ó ò! ô% öñ úõ ûõ üî °î ¢ù £| • ¶! ß% ®§ ™© ¨© ≠† Æ´ ∞Ö ≤Ø ≥ µ ∂! ∑% ∏± ∫¥ ª  Ωº Ω Ω ∆∆ «« ∆∆  ∆∆  ∆∆ † «« †´ «« ´» .» ?» P» a» r	… k	… t  Ö	À '	À 0	À ~	Ã 8	Ã A
Ã è	Õ Z	Õ c
Õ §
Œ Ø	œ 	œ 	œ 	œ !	œ #	œ %– 	— I	— R
— ñ“ ” "
rhs"
_Z13get_global_idj"
llvm.fmuladd.f64*Ü
npb-LU-rhs.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Ä
 
transfer_bytes_log1p
ﬁ}òA

wgsize
 

transfer_bytes
¯ô¿Z

wgsize_log1p
ﬁ}òA

devmap_label
