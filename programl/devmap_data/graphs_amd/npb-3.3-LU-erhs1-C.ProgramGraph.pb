

[external]
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 2) #3
4truncB+
)
	full_text

%8 = trunc i64 %7 to i32
"i64B

	full_text


i64 %7
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 1) #3
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
1%11 = tail call i64 @_Z13get_global_idj(i32 0) #3
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
%13 = icmp slt i32 %8, %5
"i32B

	full_text


i32 %8
5icmpB-
+
	full_text

%14 = icmp slt i32 %10, %4
#i32B

	full_text
	
i32 %10
/andB(
&
	full_text

%15 = and i1 %13, %14
!i1B

	full_text


i1 %13
!i1B

	full_text


i1 %14
5icmpB-
+
	full_text

%16 = icmp slt i32 %12, %3
#i32B

	full_text
	
i32 %12
/andB(
&
	full_text

%17 = and i1 %15, %16
!i1B

	full_text


i1 %15
!i1B

	full_text


i1 %16
8brB2
0
	full_text#
!
br i1 %17, label %18, label %80
!i1B

	full_text


i1 %17
Ybitcast8BL
J
	full_text=
;
9%19 = bitcast double* %0 to [163 x [163 x [5 x double]]]*
Ybitcast8BL
J
	full_text=
;
9%20 = bitcast double* %1 to [163 x [163 x [5 x double]]]*
Jbitcast8B=
;
	full_text.
,
*%21 = bitcast double* %2 to [13 x double]*
<sitofp8B0
.
	full_text!

%22 = sitofp i32 %8 to double
$i328B

	full_text


i32 %8
4add8B+
)
	full_text

%23 = add nsw i32 %5, -1
=sitofp8B1
/
	full_text"
 
%24 = sitofp i32 %23 to double
%i328B

	full_text
	
i32 %23
7fdiv8B-
+
	full_text

%25 = fdiv double %22, %24
+double8B

	full_text


double %22
+double8B

	full_text


double %24
=sitofp8B1
/
	full_text"
 
%26 = sitofp i32 %10 to double
%i328B

	full_text
	
i32 %10
@fdiv8B6
4
	full_text'
%
#%27 = fdiv double %26, 1.610000e+02
+double8B

	full_text


double %26
=sitofp8B1
/
	full_text"
 
%28 = sitofp i32 %12 to double
%i328B

	full_text
	
i32 %12
@fdiv8B6
4
	full_text'
%
#%29 = fdiv double %28, 1.610000e+02
+double8B

	full_text


double %28
0shl8B'
%
	full_text

%30 = shl i64 %7, 32
$i648B

	full_text


i64 %7
9ashr8B/
-
	full_text 

%31 = ashr exact i64 %30, 32
%i648B

	full_text
	
i64 %30
0shl8B'
%
	full_text

%32 = shl i64 %9, 32
$i648B

	full_text


i64 %9
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
%34 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%35 = ashr exact i64 %34, 32
%i648B

	full_text
	
i64 %34
'br8B

	full_text

br label %36
Bphi8B9
7
	full_text*
(
&%37 = phi i64 [ 0, %18 ], [ %78, %36 ]
%i648B

	full_text
	
i64 %78
´getelementptr8Bó
î
	full_textÜ
É
Ä%38 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %19, i64 %31, i64 %33, i64 %35, i64 %37
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %37
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %38, align 8, !tbaa !8
-double*8B

	full_text

double* %38
tgetelementptr8Ba
_
	full_textR
P
N%39 = getelementptr inbounds [13 x double], [13 x double]* %21, i64 %37, i64 0
;[13 x double]*8B%
#
	full_text

[13 x double]* %21
%i648B

	full_text
	
i64 %37
Nload8BD
B
	full_text5
3
1%40 = load double, double* %39, align 8, !tbaa !8
-double*8B

	full_text

double* %39
tgetelementptr8Ba
_
	full_textR
P
N%41 = getelementptr inbounds [13 x double], [13 x double]* %21, i64 %37, i64 1
;[13 x double]*8B%
#
	full_text

[13 x double]* %21
%i648B

	full_text
	
i64 %37
Nload8BD
B
	full_text5
3
1%42 = load double, double* %41, align 8, !tbaa !8
-double*8B

	full_text

double* %41
tgetelementptr8Ba
_
	full_textR
P
N%43 = getelementptr inbounds [13 x double], [13 x double]* %21, i64 %37, i64 4
;[13 x double]*8B%
#
	full_text

[13 x double]* %21
%i648B

	full_text
	
i64 %37
Nload8BD
B
	full_text5
3
1%44 = load double, double* %43, align 8, !tbaa !8
-double*8B

	full_text

double* %43
tgetelementptr8Ba
_
	full_textR
P
N%45 = getelementptr inbounds [13 x double], [13 x double]* %21, i64 %37, i64 7
;[13 x double]*8B%
#
	full_text

[13 x double]* %21
%i648B

	full_text
	
i64 %37
Nload8BD
B
	full_text5
3
1%46 = load double, double* %45, align 8, !tbaa !8
-double*8B

	full_text

double* %45
ugetelementptr8Bb
`
	full_textS
Q
O%47 = getelementptr inbounds [13 x double], [13 x double]* %21, i64 %37, i64 10
;[13 x double]*8B%
#
	full_text

[13 x double]* %21
%i648B

	full_text
	
i64 %37
Nload8BD
B
	full_text5
3
1%48 = load double, double* %47, align 8, !tbaa !8
-double*8B

	full_text

double* %47
icall8B_
]
	full_textP
N
L%49 = tail call double @llvm.fmuladd.f64(double %48, double %29, double %46)
+double8B

	full_text


double %48
+double8B

	full_text


double %29
+double8B

	full_text


double %46
icall8B_
]
	full_textP
N
L%50 = tail call double @llvm.fmuladd.f64(double %49, double %29, double %44)
+double8B

	full_text


double %49
+double8B

	full_text


double %29
+double8B

	full_text


double %44
icall8B_
]
	full_textP
N
L%51 = tail call double @llvm.fmuladd.f64(double %50, double %29, double %42)
+double8B

	full_text


double %50
+double8B

	full_text


double %29
+double8B

	full_text


double %42
icall8B_
]
	full_textP
N
L%52 = tail call double @llvm.fmuladd.f64(double %51, double %29, double %40)
+double8B

	full_text


double %51
+double8B

	full_text


double %29
+double8B

	full_text


double %40
tgetelementptr8Ba
_
	full_textR
P
N%53 = getelementptr inbounds [13 x double], [13 x double]* %21, i64 %37, i64 2
;[13 x double]*8B%
#
	full_text

[13 x double]* %21
%i648B

	full_text
	
i64 %37
Nload8BD
B
	full_text5
3
1%54 = load double, double* %53, align 8, !tbaa !8
-double*8B

	full_text

double* %53
tgetelementptr8Ba
_
	full_textR
P
N%55 = getelementptr inbounds [13 x double], [13 x double]* %21, i64 %37, i64 5
;[13 x double]*8B%
#
	full_text

[13 x double]* %21
%i648B

	full_text
	
i64 %37
Nload8BD
B
	full_text5
3
1%56 = load double, double* %55, align 8, !tbaa !8
-double*8B

	full_text

double* %55
tgetelementptr8Ba
_
	full_textR
P
N%57 = getelementptr inbounds [13 x double], [13 x double]* %21, i64 %37, i64 8
;[13 x double]*8B%
#
	full_text

[13 x double]* %21
%i648B

	full_text
	
i64 %37
Nload8BD
B
	full_text5
3
1%58 = load double, double* %57, align 8, !tbaa !8
-double*8B

	full_text

double* %57
ugetelementptr8Bb
`
	full_textS
Q
O%59 = getelementptr inbounds [13 x double], [13 x double]* %21, i64 %37, i64 11
;[13 x double]*8B%
#
	full_text

[13 x double]* %21
%i648B

	full_text
	
i64 %37
Nload8BD
B
	full_text5
3
1%60 = load double, double* %59, align 8, !tbaa !8
-double*8B

	full_text

double* %59
icall8B_
]
	full_textP
N
L%61 = tail call double @llvm.fmuladd.f64(double %60, double %27, double %58)
+double8B

	full_text


double %60
+double8B

	full_text


double %27
+double8B

	full_text


double %58
icall8B_
]
	full_textP
N
L%62 = tail call double @llvm.fmuladd.f64(double %61, double %27, double %56)
+double8B

	full_text


double %61
+double8B

	full_text


double %27
+double8B

	full_text


double %56
icall8B_
]
	full_textP
N
L%63 = tail call double @llvm.fmuladd.f64(double %62, double %27, double %54)
+double8B

	full_text


double %62
+double8B

	full_text


double %27
+double8B

	full_text


double %54
icall8B_
]
	full_textP
N
L%64 = tail call double @llvm.fmuladd.f64(double %63, double %27, double %52)
+double8B

	full_text


double %63
+double8B

	full_text


double %27
+double8B

	full_text


double %52
tgetelementptr8Ba
_
	full_textR
P
N%65 = getelementptr inbounds [13 x double], [13 x double]* %21, i64 %37, i64 3
;[13 x double]*8B%
#
	full_text

[13 x double]* %21
%i648B

	full_text
	
i64 %37
Nload8BD
B
	full_text5
3
1%66 = load double, double* %65, align 8, !tbaa !8
-double*8B

	full_text

double* %65
tgetelementptr8Ba
_
	full_textR
P
N%67 = getelementptr inbounds [13 x double], [13 x double]* %21, i64 %37, i64 6
;[13 x double]*8B%
#
	full_text

[13 x double]* %21
%i648B

	full_text
	
i64 %37
Nload8BD
B
	full_text5
3
1%68 = load double, double* %67, align 8, !tbaa !8
-double*8B

	full_text

double* %67
tgetelementptr8Ba
_
	full_textR
P
N%69 = getelementptr inbounds [13 x double], [13 x double]* %21, i64 %37, i64 9
;[13 x double]*8B%
#
	full_text

[13 x double]* %21
%i648B

	full_text
	
i64 %37
Nload8BD
B
	full_text5
3
1%70 = load double, double* %69, align 8, !tbaa !8
-double*8B

	full_text

double* %69
ugetelementptr8Bb
`
	full_textS
Q
O%71 = getelementptr inbounds [13 x double], [13 x double]* %21, i64 %37, i64 12
;[13 x double]*8B%
#
	full_text

[13 x double]* %21
%i648B

	full_text
	
i64 %37
Nload8BD
B
	full_text5
3
1%72 = load double, double* %71, align 8, !tbaa !8
-double*8B

	full_text

double* %71
icall8B_
]
	full_textP
N
L%73 = tail call double @llvm.fmuladd.f64(double %72, double %25, double %70)
+double8B

	full_text


double %72
+double8B

	full_text


double %25
+double8B

	full_text


double %70
icall8B_
]
	full_textP
N
L%74 = tail call double @llvm.fmuladd.f64(double %73, double %25, double %68)
+double8B

	full_text


double %73
+double8B

	full_text


double %25
+double8B

	full_text


double %68
icall8B_
]
	full_textP
N
L%75 = tail call double @llvm.fmuladd.f64(double %74, double %25, double %66)
+double8B

	full_text


double %74
+double8B

	full_text


double %25
+double8B

	full_text


double %66
icall8B_
]
	full_textP
N
L%76 = tail call double @llvm.fmuladd.f64(double %75, double %25, double %64)
+double8B

	full_text


double %75
+double8B

	full_text


double %25
+double8B

	full_text


double %64
´getelementptr8Bó
î
	full_textÜ
É
Ä%77 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %20, i64 %31, i64 %33, i64 %35, i64 %37
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %20
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %37
Nstore8BC
A
	full_text4
2
0store double %76, double* %77, align 8, !tbaa !8
+double8B

	full_text


double %76
-double*8B

	full_text

double* %77
8add8B/
-
	full_text 

%78 = add nuw nsw i64 %37, 1
%i648B

	full_text
	
i64 %37
5icmp8B+
)
	full_text

%79 = icmp eq i64 %78, 5
%i648B

	full_text
	
i64 %78
:br8B2
0
	full_text#
!
br i1 %79, label %80, label %36
#i18B

	full_text


i1 %79
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %2
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %1
$i328B
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
$i648B

	full_text


i64 11
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 2
$i648B

	full_text


i64 10
$i648B

	full_text


i64 12
4double8B&
$
	full_text

double 0.000000e+00
#i648B

	full_text	

i64 8
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 4
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 5
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 6
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 7
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 9
4double8B&
$
	full_text

double 1.610000e+02
#i648B

	full_text	

i64 2       	  
 

                     !  "    #$ ## %& %% '( '' )* )) +, ++ -. -- /0 // 12 11 34 33 56 55 79 88 :; :< := :> :? :: @A @@ BC BD BB EF EE GH GI GG JK JJ LM LN LL OP OO QR QS QQ TU TT VW VX VV YZ YY [\ [] [^ [[ _` _a _b __ cd ce cf cc gh gi gj gg kl km kk no nn pq pr pp st ss uv uw uu xy xx z{ z| zz }~ }} Ä 	Å 	Ç  ÉÑ É
Ö É
Ü ÉÉ áà á
â á
ä áá ãå ã
ç ã
é ãã èê è
ë èè íì íí îï î
ñ îî óò óó ôö ô
õ ôô úù úú ûü û
† ûû °¢ °° £§ £
• £
¶ ££ ß® ß
© ß
™ ßß ´¨ ´
≠ ´
Æ ´´ Ø∞ Ø
± Ø
≤ ØØ ≥¥ ≥
µ ≥
∂ ≥
∑ ≥
∏ ≥≥ π∫ π
ª ππ ºΩ ºº æø ææ ¿¡ ¿	√ 
√ 	ƒ ≈ ∆ « 	»    	  
         ! " $# & (' * ,+ . 0/ 2 43 6º 9 ;- <1 =5 >8 ?: A C8 DB F H8 IG K M8 NL P R8 SQ U W8 XV ZY \) ]T ^[ `) aO b_ d) eJ fc h) iE j l8 mk o q8 rp t v8 wu y {8 |z ~} Ä% Åx Ç Ñ% Ös ÜÉ à% ân äá å% çg é ê8 ëè ì ï8 ñî ò ö8 õô ù ü8 †û ¢° §  •ú ¶£ ®  ©ó ™ß ¨  ≠í Æ´ ∞  ±ã ≤ ¥- µ1 ∂5 ∑8 ∏Ø ∫≥ ª8 Ωº øæ ¡  ¬7 8¿ ¬¿ 8 ……    ¬ …… á    áß    ß_    _É    Éã    ã£    £ …… c    c …… Ø    Ø[    [    g    g´    ´	À zÃ Õ 	Œ V
œ û– @	— u
“ è	” L	‘ G
‘ º	’ p
’ æ	÷ +	÷ -	÷ /	÷ 1	÷ 3	÷ 5
◊ î	ÿ Ÿ 8	Ÿ B	⁄ Q€ 
‹ ô	› %	› )	ﬁ k"
erhs1"
_Z13get_global_idj"
llvm.fmuladd.f64*à
npb-LU-erhs1.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å
 
transfer_bytes_log1p
	å£A

wgsize_log1p
	å£A

devmap_label


wgsize
@

transfer_bytes	
ÿ√∂Ë