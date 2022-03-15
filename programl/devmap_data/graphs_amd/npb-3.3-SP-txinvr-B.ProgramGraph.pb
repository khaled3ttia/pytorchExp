
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
br i1 %24, label %89, label %25
!i1B

	full_text


i1 %24
Sbitcast8BF
D
	full_text7
5
3%26 = bitcast double* %0 to [103 x [103 x double]]*
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
k%39 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %30, i64 %34, i64 %36, i64 %38
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
1%40 = load double, double* %39, align 8, !tbaa !8
-double*8B

	full_text

double* %39
‘getelementptr8B~
|
	full_texto
m
k%41 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %26, i64 %34, i64 %36, i64 %38
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %26
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
k%43 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %27, i64 %34, i64 %36, i64 %38
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
1%44 = load double, double* %43, align 8, !tbaa !8
-double*8B

	full_text

double* %43
‘getelementptr8B~
|
	full_texto
m
k%45 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %28, i64 %34, i64 %36, i64 %38
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
1%46 = load double, double* %45, align 8, !tbaa !8
-double*8B

	full_text

double* %45
‘getelementptr8B~
|
	full_texto
m
k%47 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %31, i64 %34, i64 %36, i64 %38
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
1%48 = load double, double* %47, align 8, !tbaa !8
-double*8B

	full_text

double* %47
7fmul8B-
+
	full_text

%49 = fmul double %48, %48
+double8B

	full_text


double %48
+double8B

	full_text


double %48
¨getelementptr8B”
‘
	full_textƒ
€
~%50 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %32, i64 %34, i64 %36, i64 %38, i64 0
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
~%52 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %32, i64 %34, i64 %36, i64 %38, i64 1
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
~%54 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %32, i64 %34, i64 %36, i64 %38, i64 2
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
~%56 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %32, i64 %34, i64 %36, i64 %38, i64 3
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
~%58 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %32, i64 %34, i64 %36, i64 %38, i64 4
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
1%59 = load double, double* %58, align 8, !tbaa !8
-double*8B

	full_text

double* %58
@fdiv8B6
4
	full_text'
%
#%60 = fdiv double 4.000000e-01, %49
+double8B

	full_text


double %49
‘getelementptr8B~
|
	full_texto
m
k%61 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %29, i64 %34, i64 %36, i64 %38
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
1%62 = load double, double* %61, align 8, !tbaa !8
-double*8B

	full_text

double* %61
7fmul8B-
+
	full_text

%63 = fmul double %42, %53
+double8B

	full_text


double %42
+double8B

	full_text


double %53
Afsub8B7
5
	full_text(
&
$%64 = fsub double -0.000000e+00, %63
+double8B

	full_text


double %63
icall8B_
]
	full_textP
N
L%65 = tail call double @llvm.fmuladd.f64(double %62, double %51, double %64)
+double8B

	full_text


double %62
+double8B

	full_text


double %51
+double8B

	full_text


double %64
Afsub8B7
5
	full_text(
&
$%66 = fsub double -0.000000e+00, %44
+double8B

	full_text


double %44
icall8B_
]
	full_textP
N
L%67 = tail call double @llvm.fmuladd.f64(double %66, double %55, double %65)
+double8B

	full_text


double %66
+double8B

	full_text


double %55
+double8B

	full_text


double %65
Afsub8B7
5
	full_text(
&
$%68 = fsub double -0.000000e+00, %46
+double8B

	full_text


double %46
icall8B_
]
	full_textP
N
L%69 = tail call double @llvm.fmuladd.f64(double %68, double %57, double %67)
+double8B

	full_text


double %68
+double8B

	full_text


double %57
+double8B

	full_text


double %67
7fadd8B-
+
	full_text

%70 = fadd double %59, %69
+double8B

	full_text


double %59
+double8B

	full_text


double %69
7fmul8B-
+
	full_text

%71 = fmul double %60, %70
+double8B

	full_text


double %60
+double8B

	full_text


double %70
Ucall8BK
I
	full_text<
:
8%72 = tail call double @_Z4sqrtd(double 5.000000e-01) #3
7fmul8B-
+
	full_text

%73 = fmul double %40, %72
+double8B

	full_text


double %40
+double8B

	full_text


double %72
Afsub8B7
5
	full_text(
&
$%74 = fsub double -0.000000e+00, %53
+double8B

	full_text


double %53
icall8B_
]
	full_textP
N
L%75 = tail call double @llvm.fmuladd.f64(double %42, double %51, double %74)
+double8B

	full_text


double %42
+double8B

	full_text


double %51
+double8B

	full_text


double %74
7fmul8B-
+
	full_text

%76 = fmul double %75, %73
+double8B

	full_text


double %75
+double8B

	full_text


double %73
7fmul8B-
+
	full_text

%77 = fmul double %48, %73
+double8B

	full_text


double %48
+double8B

	full_text


double %73
7fmul8B-
+
	full_text

%78 = fmul double %77, %71
+double8B

	full_text


double %77
+double8B

	full_text


double %71
7fsub8B-
+
	full_text

%79 = fsub double %51, %71
+double8B

	full_text


double %51
+double8B

	full_text


double %71
Nstore8BC
A
	full_text4
2
0store double %79, double* %50, align 8, !tbaa !8
+double8B

	full_text


double %79
-double*8B

	full_text

double* %50
Afsub8B7
5
	full_text(
&
$%80 = fsub double -0.000000e+00, %57
+double8B

	full_text


double %57
icall8B_
]
	full_textP
N
L%81 = tail call double @llvm.fmuladd.f64(double %46, double %51, double %80)
+double8B

	full_text


double %46
+double8B

	full_text


double %51
+double8B

	full_text


double %80
7fmul8B-
+
	full_text

%82 = fmul double %40, %81
+double8B

	full_text


double %40
+double8B

	full_text


double %81
Afsub8B7
5
	full_text(
&
$%83 = fsub double -0.000000e+00, %82
+double8B

	full_text


double %82
Nstore8BC
A
	full_text4
2
0store double %83, double* %52, align 8, !tbaa !8
+double8B

	full_text


double %83
-double*8B

	full_text

double* %52
Afsub8B7
5
	full_text(
&
$%84 = fsub double -0.000000e+00, %55
+double8B

	full_text


double %55
icall8B_
]
	full_textP
N
L%85 = tail call double @llvm.fmuladd.f64(double %44, double %51, double %84)
+double8B

	full_text


double %44
+double8B

	full_text


double %51
+double8B

	full_text


double %84
7fmul8B-
+
	full_text

%86 = fmul double %40, %85
+double8B

	full_text


double %40
+double8B

	full_text


double %85
Nstore8BC
A
	full_text4
2
0store double %86, double* %54, align 8, !tbaa !8
+double8B

	full_text


double %86
-double*8B

	full_text

double* %54
7fsub8B-
+
	full_text

%87 = fsub double %78, %76
+double8B

	full_text


double %78
+double8B

	full_text


double %76
Nstore8BC
A
	full_text4
2
0store double %87, double* %56, align 8, !tbaa !8
+double8B

	full_text


double %87
-double*8B

	full_text

double* %56
7fadd8B-
+
	full_text

%88 = fadd double %76, %78
+double8B

	full_text


double %76
+double8B

	full_text


double %78
Nstore8BC
A
	full_text4
2
0store double %88, double* %58, align 8, !tbaa !8
+double8B

	full_text


double %88
-double*8B

	full_text

double* %58
'br8B

	full_text

br label %89
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %7
,double*8B

	full_text


double* %6
$i328B

	full_text


i32 %8
,double*8B

	full_text


double* %5
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
i32 %9
,double*8B

	full_text


double* %2
,double*8B

	full_text


double* %3
,double*8B

	full_text


double* %4
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
i64 0
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
i64 4
4double8B&
$
	full_text

double 4.000000e-01
#i648B

	full_text	

i64 1
5double8B'
%
	full_text

double -0.000000e+00
#i328B

	full_text	

i32 1        		 
 

                       !! "" ## $$ %& %% '( '' )* )) +, ++ -. -- /0 // 12 13 14 15 11 67 66 89 8: 8; 8< 88 => == ?@ ?A ?B ?C ?? DE DD FG FH FI FJ FF KL KK MN MO MP MQ MM RS RR TU TV TT WX WY WZ W[ WW \] \\ ^_ ^` ^a ^b ^^ cd cc ef eg eh ei ee jk jj lm ln lo lp ll qr qq st su sv sw ss xy xx z{ zz |} |~ | |	€ || ‚  ƒ„ ƒ
… ƒƒ †
‡ †† ‰ 
 
‹  
   
 
‘  ’
“ ’’ ”• ”
– ”
— ”” ™ 
  › ›
 ››    
΅  Ά
£ ΆΆ ¤¥ ¤
¦ ¤
§ ¤¤ ¨© ¨
ª ¨¨ «¬ «
­ «« ®― ®
° ®® ±² ±
³ ±± ΄µ ΄
¶ ΄΄ ·
Έ ·· ΉΊ Ή
» Ή
Ό ΉΉ ½Ύ ½
Ώ ½½ ΐ
Α ΐΐ ΒΓ Β
Δ ΒΒ Ε
Ζ ΕΕ ΗΘ Η
Ι Η
Κ ΗΗ ΛΜ Λ
Ν ΛΛ ΞΟ Ξ
Π ΞΞ ΡÒ Ρ
Σ ΡΡ ΤΥ Τ
Φ ΤΤ ΧΨ Χ
Ω ΧΧ ΪΫ Ϊ
ά ΪΪ έ	ί ΰ $	α β #γ δ 	ε ζ  η !θ "   	 
           &% ( *) ,
 .- 0" 2' 3+ 4/ 51 7 9' :+ ;/ <8 > @' A+ B/ C? E  G' H+ I/ JF L# N' O+ P/ QM SR UR V$ X' Y+ Z/ [W ]$ _' `+ a/ b^ d$ f' g+ h/ ie k$ m' n+ o/ pl r$ t' u+ v/ ws yT {! }' ~+ / €| ‚= „c …ƒ ‡ ‰\ † ‹D  j  ‘K “’ •q – —x ™” z  6   ΅c £= ¥\ ¦Ά §¤ © ªR ¬ ­« ―› °\ ²› ³± µW ¶q ΈK Ί\ »· Ό6 ΎΉ Ώ½ Αΐ Γ^ Δj ΖD Θ\ ΙΕ Κ6 ΜΗ ΝΛ Οe Π® Ò¨ ΣΡ Υl Φ¨ Ψ® ΩΧ Ϋs ά ή έ ή ή ιι λλ κκ ιι ” κκ ” κκ ¤ κκ ¤ κκ Ή κκ Ή ιι  λλ Η κκ Η	 ιι 		μ lν 	ξ %	ξ '	ξ )	ξ +	ξ -	ξ /ο 		π W	ρ eς 	σ sτ z	υ 	υ 	υ 
	υ ^φ †φ φ ’φ Άφ ·φ ΐφ Εχ "
txinvr"
_Z13get_global_idj"
llvm.fmuladd.f64"

_Z4sqrtd*‰
npb-SP-txinvr.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02

wgsize_log1p
|A

devmap_label


wgsize
 

transfer_bytes	
θυσΨ
 
transfer_bytes_log1p
|A