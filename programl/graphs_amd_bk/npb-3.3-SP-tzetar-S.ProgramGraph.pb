

[external]
LcallBD
B
	full_text5
3
1%11 = tail call i64 @_Z13get_global_idj(i32 2) #3
.addB'
%
	full_text

%12 = add i64 %11, 1
#i64B

	full_text
	
i64 %11
6truncB-
+
	full_text

%13 = trunc i64 %12 to i32
#i64B

	full_text
	
i64 %12
LcallBD
B
	full_text5
3
1%14 = tail call i64 @_Z13get_global_idj(i32 1) #3
.addB'
%
	full_text

%15 = add i64 %14, 1
#i64B

	full_text
	
i64 %14
LcallBD
B
	full_text5
3
1%16 = tail call i64 @_Z13get_global_idj(i32 0) #3
.addB'
%
	full_text

%17 = add i64 %16, 1
#i64B

	full_text
	
i64 %16
6truncB-
+
	full_text

%18 = trunc i64 %17 to i32
#i64B

	full_text
	
i64 %17
5icmpB-
+
	full_text

%19 = icmp sgt i32 %13, %9
#i32B

	full_text
	
i32 %13
6truncB-
+
	full_text

%20 = trunc i64 %15 to i32
#i64B

	full_text
	
i64 %15
5icmpB-
+
	full_text

%21 = icmp sgt i32 %20, %8
#i32B

	full_text
	
i32 %20
-orB'
%
	full_text

%22 = or i1 %19, %21
!i1B

	full_text


i1 %19
!i1B

	full_text


i1 %21
5icmpB-
+
	full_text

%23 = icmp sgt i32 %18, %7
#i32B

	full_text
	
i32 %18
-orB'
%
	full_text

%24 = or i1 %22, %23
!i1B

	full_text


i1 %22
!i1B

	full_text


i1 %23
8brB2
0
	full_text#
!
br i1 %24, label %84, label %25
!i1B

	full_text


i1 %24
Wbitcast8BJ
H
	full_text;
9
7%26 = bitcast double* %0 to [13 x [13 x [5 x double]]]*
Qbitcast8BD
B
	full_text5
3
1%27 = bitcast double* %1 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%28 = bitcast double* %2 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%29 = bitcast double* %3 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%30 = bitcast double* %4 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%31 = bitcast double* %5 to [13 x [13 x double]]*
Wbitcast8BJ
H
	full_text;
9
7%32 = bitcast double* %6 to [13 x [13 x [5 x double]]]*
1shl8B(
&
	full_text

%33 = shl i64 %12, 32
%i648B

	full_text
	
i64 %12
9ashr8B/
-
	full_text 

%34 = ashr exact i64 %33, 32
%i648B

	full_text
	
i64 %33
1shl8B(
&
	full_text

%35 = shl i64 %15, 32
%i648B

	full_text
	
i64 %15
9ashr8B/
-
	full_text 

%36 = ashr exact i64 %35, 32
%i648B

	full_text
	
i64 %35
1shl8B(
&
	full_text

%37 = shl i64 %17, 32
%i648B

	full_text
	
i64 %17
9ashr8B/
-
	full_text 

%38 = ashr exact i64 %37, 32
%i648B

	full_text
	
i64 %37
çgetelementptr8Bz
x
	full_textk
i
g%39 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %27, i64 %34, i64 %36, i64 %38
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %27
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %36
%i648B

	full_text
	
i64 %38
Nload8BD
B
	full_text5
3
1%40 = load double, double* %39, align 8, !tbaa !8
-double*8B

	full_text

double* %39
çgetelementptr8Bz
x
	full_textk
i
g%41 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %28, i64 %34, i64 %36, i64 %38
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %28
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %36
%i648B

	full_text
	
i64 %38
Nload8BD
B
	full_text5
3
1%42 = load double, double* %41, align 8, !tbaa !8
-double*8B

	full_text

double* %41
çgetelementptr8Bz
x
	full_textk
i
g%43 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %29, i64 %34, i64 %36, i64 %38
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %29
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %36
%i648B

	full_text
	
i64 %38
Nload8BD
B
	full_text5
3
1%44 = load double, double* %43, align 8, !tbaa !8
-double*8B

	full_text

double* %43
çgetelementptr8Bz
x
	full_textk
i
g%45 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %31, i64 %34, i64 %36, i64 %38
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %31
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %36
%i648B

	full_text
	
i64 %38
Nload8BD
B
	full_text5
3
1%46 = load double, double* %45, align 8, !tbaa !8
-double*8B

	full_text

double* %45
7fmul8B-
+
	full_text

%47 = fmul double %46, %46
+double8B

	full_text


double %46
+double8B

	full_text


double %46
¢getelementptr8Bé
ã
	full_text~
|
z%48 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %32, i64 %34, i64 %36, i64 %38, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %36
%i648B

	full_text
	
i64 %38
Nload8BD
B
	full_text5
3
1%49 = load double, double* %48, align 8, !tbaa !8
-double*8B

	full_text

double* %48
¢getelementptr8Bé
ã
	full_text~
|
z%50 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %32, i64 %34, i64 %36, i64 %38, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %36
%i648B

	full_text
	
i64 %38
Nload8BD
B
	full_text5
3
1%51 = load double, double* %50, align 8, !tbaa !8
-double*8B

	full_text

double* %50
¢getelementptr8Bé
ã
	full_text~
|
z%52 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %32, i64 %34, i64 %36, i64 %38, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %36
%i648B

	full_text
	
i64 %38
Nload8BD
B
	full_text5
3
1%53 = load double, double* %52, align 8, !tbaa !8
-double*8B

	full_text

double* %52
¢getelementptr8Bé
ã
	full_text~
|
z%54 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %32, i64 %34, i64 %36, i64 %38, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %36
%i648B

	full_text
	
i64 %38
Nload8BD
B
	full_text5
3
1%55 = load double, double* %54, align 8, !tbaa !8
-double*8B

	full_text

double* %54
¢getelementptr8Bé
ã
	full_text~
|
z%56 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %32, i64 %34, i64 %36, i64 %38, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %36
%i648B

	full_text
	
i64 %38
Nload8BD
B
	full_text5
3
1%57 = load double, double* %56, align 8, !tbaa !8
-double*8B

	full_text

double* %56
¢getelementptr8Bé
ã
	full_text~
|
z%58 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %26, i64 %34, i64 %36, i64 %38, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %36
%i648B

	full_text
	
i64 %38
Nload8BD
B
	full_text5
3
1%59 = load double, double* %58, align 8, !tbaa !8
-double*8B

	full_text

double* %58
Ucall8BK
I
	full_text<
:
8%60 = tail call double @_Z4sqrtd(double 5.000000e-01) #3
7fmul8B-
+
	full_text

%61 = fmul double %59, %60
+double8B

	full_text


double %59
+double8B

	full_text


double %60
7fdiv8B-
+
	full_text

%62 = fdiv double %61, %46
+double8B

	full_text


double %61
+double8B

	full_text


double %46
7fadd8B-
+
	full_text

%63 = fadd double %55, %57
+double8B

	full_text


double %55
+double8B

	full_text


double %57
7fmul8B-
+
	full_text

%64 = fmul double %63, %62
+double8B

	full_text


double %63
+double8B

	full_text


double %62
7fadd8B-
+
	full_text

%65 = fadd double %53, %64
+double8B

	full_text


double %53
+double8B

	full_text


double %64
7fsub8B-
+
	full_text

%66 = fsub double %55, %57
+double8B

	full_text


double %55
+double8B

	full_text


double %57
7fmul8B-
+
	full_text

%67 = fmul double %66, %61
+double8B

	full_text


double %66
+double8B

	full_text


double %61
Nstore8BC
A
	full_text4
2
0store double %65, double* %48, align 8, !tbaa !8
+double8B

	full_text


double %65
-double*8B

	full_text

double* %48
Afsub8B7
5
	full_text(
&
$%68 = fsub double -0.000000e+00, %59
+double8B

	full_text


double %59
7fmul8B-
+
	full_text

%69 = fmul double %40, %65
+double8B

	full_text


double %40
+double8B

	full_text


double %65
icall8B_
]
	full_textP
N
L%70 = tail call double @llvm.fmuladd.f64(double %68, double %51, double %69)
+double8B

	full_text


double %68
+double8B

	full_text


double %51
+double8B

	full_text


double %69
Nstore8BC
A
	full_text4
2
0store double %70, double* %50, align 8, !tbaa !8
+double8B

	full_text


double %70
-double*8B

	full_text

double* %50
7fmul8B-
+
	full_text

%71 = fmul double %42, %65
+double8B

	full_text


double %42
+double8B

	full_text


double %65
icall8B_
]
	full_textP
N
L%72 = tail call double @llvm.fmuladd.f64(double %59, double %49, double %71)
+double8B

	full_text


double %59
+double8B

	full_text


double %49
+double8B

	full_text


double %71
Nstore8BC
A
	full_text4
2
0store double %72, double* %52, align 8, !tbaa !8
+double8B

	full_text


double %72
-double*8B

	full_text

double* %52
icall8B_
]
	full_textP
N
L%73 = tail call double @llvm.fmuladd.f64(double %44, double %65, double %67)
+double8B

	full_text


double %44
+double8B

	full_text


double %65
+double8B

	full_text


double %67
Nstore8BC
A
	full_text4
2
0store double %73, double* %54, align 8, !tbaa !8
+double8B

	full_text


double %73
-double*8B

	full_text

double* %54
Afsub8B7
5
	full_text(
&
$%74 = fsub double -0.000000e+00, %40
+double8B

	full_text


double %40
7fmul8B-
+
	full_text

%75 = fmul double %42, %49
+double8B

	full_text


double %42
+double8B

	full_text


double %49
icall8B_
]
	full_textP
N
L%76 = tail call double @llvm.fmuladd.f64(double %74, double %51, double %75)
+double8B

	full_text


double %74
+double8B

	full_text


double %51
+double8B

	full_text


double %75
çgetelementptr8Bz
x
	full_textk
i
g%77 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %30, i64 %34, i64 %36, i64 %38
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %30
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %36
%i648B

	full_text
	
i64 %38
Nload8BD
B
	full_text5
3
1%78 = load double, double* %77, align 8, !tbaa !8
-double*8B

	full_text

double* %77
7fmul8B-
+
	full_text

%79 = fmul double %65, %78
+double8B

	full_text


double %65
+double8B

	full_text


double %78
icall8B_
]
	full_textP
N
L%80 = tail call double @llvm.fmuladd.f64(double %59, double %76, double %79)
+double8B

	full_text


double %59
+double8B

	full_text


double %76
+double8B

	full_text


double %79
@fmul8B6
4
	full_text'
%
#%81 = fmul double %47, 2.500000e+00
+double8B

	full_text


double %47
icall8B_
]
	full_textP
N
L%82 = tail call double @llvm.fmuladd.f64(double %81, double %64, double %80)
+double8B

	full_text


double %81
+double8B

	full_text


double %64
+double8B

	full_text


double %80
icall8B_
]
	full_textP
N
L%83 = tail call double @llvm.fmuladd.f64(double %44, double %67, double %82)
+double8B

	full_text


double %44
+double8B

	full_text


double %67
+double8B

	full_text


double %82
Nstore8BC
A
	full_text4
2
0store double %83, double* %56, align 8, !tbaa !8
+double8B

	full_text


double %83
-double*8B

	full_text

double* %56
'br8B

	full_text

br label %84
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


double* %0
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %2
$i328B

	full_text


i32 %7
$i328B

	full_text


i32 %8
$i328B

	full_text


i32 %9
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


double* %5
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
-; undefined function B

	full_text

 
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
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 4
4double8B&
$
	full_text

double 5.000000e-01
#i648B

	full_text	

i64 3
5double8B'
%
	full_text

double -0.000000e+00
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 1
4double8B&
$
	full_text

double 2.500000e+00        		 
 

                       !! "" ## $$ %& %% '( '' )* )) +, ++ -. -- /0 // 12 13 14 15 11 67 66 89 8: 8; 8< 88 => == ?@ ?A ?B ?C ?? DE DD FG FH FI FJ FF KL KK MN MO MM PQ PR PS PT PP UV UU WX WY WZ W[ WW \] \\ ^_ ^` ^a ^b ^^ cd cc ef eg eh ei ee jk jj lm ln lo lp ll qr qq st su sv sw ss xy xx zz {| {} {{ ~ ~	Ä ~~ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü ÑÑ áà á
â áá äã ä
å ää çé ç
è çç êë ê
í êê ì
î ìì ïñ ï
ó ïï òô ò
ö ò
õ òò úù ú
û úú ü† ü
° üü ¢£ ¢
§ ¢
• ¢¢ ¶ß ¶
® ¶¶ ©™ ©
´ ©
¨ ©© ≠Æ ≠
Ø ≠≠ ∞
± ∞∞ ≤≥ ≤
¥ ≤≤ µ∂ µ
∑ µ
∏ µµ π∫ π
ª π
º π
Ω ππ æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √
∆ √√ «» «« …  …
À …
Ã …… ÕŒ Õ
œ Õ
– ÕÕ —“ —
” —— ‘÷ !◊ ÿ Ÿ  	⁄ 	€ 	‹ › $ﬁ "ﬂ #   	 
           &% ( *) ,
 .- 0 2' 3+ 4/ 51 7  9' :+ ;/ <8 >! @' A+ B/ C? E# G' H+ I/ JF LK NK O$ Q' R+ S/ TP V$ X' Y+ Z/ [W ]$ _' `+ a/ b^ d$ f' g+ h/ ie k$ m' n+ o/ pl r t' u+ v/ ws yx |z }{ K Äj Çq ÉÅ Ö~ Üc àÑ âj ãq åä é{ èá ëP íx î6 ñá óì ô\ öï õò ùW û= †á °x £U §ü •¢ ß^ ®D ™á ´ç ¨© Æe Ø6 ±= ≥U ¥∞ ∂\ ∑≤ ∏" ∫' ª+ º/ Ωπ øá ¡æ ¬x ƒµ ≈¿ ∆M »«  Ñ À√ ÃD Œç œ… –Õ “l ” ’ ‘ ’ ‚‚ ’ ‡‡ ·· ‡‡ √ ‚‚ √… ‚‚ … ‡‡ 	 ‡‡ 	ò ‚‚ ò© ‚‚ ©Õ ‚‚ Õ¢ ‚‚ ¢z ·· zµ ‚‚ µ	„ ^‰ 	Â P	Â sÊ 	Á lË z	È eÍ ìÍ ∞Î 		Ï %	Ï '	Ï )	Ï +	Ï -	Ï /	Ì 	Ì 	Ì 
	Ì W
Ó «"
tzetar"
_Z13get_global_idj"

_Z4sqrtd"
llvm.fmuladd.f64*â
npb-SP-tzetar.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02

wgsize_log1p
ãèYA
 
transfer_bytes_log1p
ãèYA

devmap_label
 

wgsize
<

transfer_bytes
∏ä1