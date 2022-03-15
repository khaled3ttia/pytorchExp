
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
Ybitcast8BL
J
	full_text=
;
9%26 = bitcast double* %0 to [103 x [103 x [5 x double]]]*
Sbitcast8BF
D
	full_text7
5
3%27 = bitcast double* %1 to [103 x [103 x double]]*
Sbitcast8BF
D
	full_text7
5
3%28 = bitcast double* %2 to [103 x [103 x double]]*
Sbitcast8BF
D
	full_text7
5
3%29 = bitcast double* %3 to [103 x [103 x double]]*
Sbitcast8BF
D
	full_text7
5
3%30 = bitcast double* %4 to [103 x [103 x double]]*
Sbitcast8BF
D
	full_text7
5
3%31 = bitcast double* %5 to [103 x [103 x double]]*
Ybitcast8BL
J
	full_text=
;
9%32 = bitcast double* %6 to [103 x [103 x [5 x double]]]*
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
‘getelementptr8B~
|
	full_texto
m
k%39 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %27, i64 %34, i64 %36, i64 %38
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %27
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
‘getelementptr8B~
|
	full_texto
m
k%41 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %28, i64 %34, i64 %36, i64 %38
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %28
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
‘getelementptr8B~
|
	full_texto
m
k%43 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %29, i64 %34, i64 %36, i64 %38
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %29
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
‘getelementptr8B~
|
	full_texto
m
k%45 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %31, i64 %34, i64 %36, i64 %38
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %31
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
¨getelementptr8B”
‘
	full_textƒ
€
~%48 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %32, i64 %34, i64 %36, i64 %38, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %32
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
¨getelementptr8B”
‘
	full_textƒ
€
~%50 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %32, i64 %34, i64 %36, i64 %38, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %32
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
¨getelementptr8B”
‘
	full_textƒ
€
~%52 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %32, i64 %34, i64 %36, i64 %38, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %32
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
¨getelementptr8B”
‘
	full_textƒ
€
~%54 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %32, i64 %34, i64 %36, i64 %38, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %32
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
¨getelementptr8B”
‘
	full_textƒ
€
~%56 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %32, i64 %34, i64 %36, i64 %38, i64 4
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %32
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
¨getelementptr8B”
‘
	full_textƒ
€
~%58 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %26, i64 %34, i64 %36, i64 %38, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %26
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
‘getelementptr8B~
|
	full_texto
m
k%77 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %30, i64 %34, i64 %36, i64 %38
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %30
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


double* %2
$i328B

	full_text


i32 %7
,double*8B

	full_text


double* %3
,double*8B

	full_text


double* %6
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %4
,double*8B

	full_text


double* %5
,double*8B

	full_text


double* %1
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
4double8B&
$
	full_text

double 5.000000e-01
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
#i328B

	full_text	

i32 2
#i648B

	full_text	

i64 2
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
4double8B&
$
	full_text

double 2.500000e+00
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 4
#i328B

	full_text	

i32 1
5double8B'
%
	full_text

double -0.000000e+00        		 
 

                       !! "" ## $$ %& %% '( '' )* )) +, ++ -. -- /0 // 12 13 14 15 11 67 66 89 8: 8; 8< 88 => == ?@ ?A ?B ?C ?? DE DD FG FH FI FJ FF KL KK MN MO MM PQ PR PS PT PP UV UU WX WY WZ W[ WW \] \\ ^_ ^` ^a ^b ^^ cd cc ef eg eh ei ee jk jj lm ln lo lp ll qr qq st su sv sw ss xy xx zz {| {} {{ ~ ~	€ ~~ ‚ 
ƒ  „… „
† „„ ‡ ‡
‰ ‡‡ ‹ 
   
  ‘ 
’  “
” ““ •– •
— •• ™ 
 
›   
    
΅  Ά£ Ά
¤ Ά
¥ ΆΆ ¦§ ¦
¨ ¦¦ ©ª ©
« ©
¬ ©© ­® ­
― ­­ °
± °° ²³ ²
΄ ²² µ¶ µ
· µ
Έ µµ ΉΊ Ή
» Ή
Ό Ή
½ ΉΉ ΎΏ ΎΎ ΐΑ ΐ
Β ΐΐ ΓΔ Γ
Ε Γ
Ζ ΓΓ ΗΘ ΗΗ ΙΚ Ι
Λ Ι
Μ ΙΙ ΝΞ Ν
Ο Ν
Π ΝΝ ΡÒ Ρ
Σ ΡΡ ΤΦ  	Χ Ψ !Ω $Ϊ Ϋ "ά #έ 	ή 	ί    	 
           &% ( *) ,
 .- 0 2' 3+ 4/ 51 7  9' :+ ;/ <8 >! @' A+ B/ C? E# G' H+ I/ JF LK NK O$ Q' R+ S/ TP V$ X' Y+ Z/ [W ]$ _' `+ a/ b^ d$ f' g+ h/ ie k$ m' n+ o/ pl r t' u+ v/ ws yx |z }{ K €j ‚q ƒ …~ †c „ ‰j ‹q  { ‡ ‘P ’x ”6 –‡ —“ ™\ • › W =  ‡ ΅x £U ¤ ¥Ά §^ ¨D ª‡ « ¬© ®e ―6 ±= ³U ΄° ¶\ ·² Έ" Ί' »+ Ό/ ½Ή Ώ‡ ΑΎ Βx Δµ Εΐ ΖM ΘΗ Κ„ ΛΓ ΜD Ξ ΟΙ ΠΝ Òl Σ Υ Τ Υ ΰΰ αα Υ ββΆ ββ Άµ ββ µΝ ββ Ν ΰΰ Γ ββ ΓΙ ββ Ι© ββ ©z αα z ββ 	 ΰΰ 	 ΰΰ γ z	δ 	δ 	δ 
	δ W	ε P	ε sζ 	η ^	θ %	θ '	θ )	θ +	θ -	θ /ι 	
κ Η	λ e	μ lν ξ “ξ °"
tzetar"
_Z13get_global_idj"

_Z4sqrtd"
llvm.fmuladd.f64*‰
npb-SP-tzetar.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02

devmap_label


transfer_bytes	
θυσΨ

wgsize
 

wgsize_log1p
|A
 
transfer_bytes_log1p
|A