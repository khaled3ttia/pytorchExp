

[external]
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 1) #3
3zextB+
)
	full_text

%11 = zext i32 %7 to i64
0addB)
'
	full_text

%12 = add i64 %10, %11
#i64B

	full_text
	
i64 %10
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
1%14 = tail call i64 @_Z13get_global_idj(i32 0) #3
3zextB+
)
	full_text

%15 = zext i32 %3 to i64
0addB)
'
	full_text

%16 = add i64 %14, %15
#i64B

	full_text
	
i64 %14
#i64B

	full_text
	
i64 %15
5icmpB-
+
	full_text

%17 = icmp slt i32 %13, %8
#i32B

	full_text
	
i32 %13
6truncB-
+
	full_text

%18 = trunc i64 %16 to i32
#i64B

	full_text
	
i64 %16
5icmpB-
+
	full_text

%19 = icmp slt i32 %18, %4
#i32B

	full_text
	
i32 %18
/andB(
&
	full_text

%20 = and i1 %17, %19
!i1B

	full_text


i1 %17
!i1B

	full_text


i1 %19
8brB2
0
	full_text#
!
br i1 %20, label %21, label %68
!i1B

	full_text


i1 %20
Wbitcast8BJ
H
	full_text;
9
7%22 = bitcast double* %0 to [33 x [33 x [5 x double]]]*
Jbitcast8B=
;
	full_text.
,
*%23 = bitcast double* %1 to [35 x double]*
Jbitcast8B=
;
	full_text.
,
*%24 = bitcast double* %2 to [35 x double]*
1shl8B(
&
	full_text

%25 = shl i64 %12, 32
%i648B

	full_text
	
i64 %12
9ashr8B/
-
	full_text 

%26 = ashr exact i64 %25, 32
%i648B

	full_text
	
i64 %25
5sext8B+
)
	full_text

%27 = sext i32 %5 to i64
1shl8B(
&
	full_text

%28 = shl i64 %16, 32
%i648B

	full_text
	
i64 %16
9ashr8B/
-
	full_text 

%29 = ashr exact i64 %28, 32
%i648B

	full_text
	
i64 %28
¢getelementptr8Bé
ã
	full_text~
|
z%30 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %22, i64 %26, i64 %27, i64 %29, i64 4
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %26
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
¢getelementptr8Bé
ã
	full_text~
|
z%32 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %22, i64 %26, i64 %27, i64 %29, i64 1
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %26
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
1%33 = load double, double* %32, align 8, !tbaa !8
-double*8B

	full_text

double* %32
¢getelementptr8Bé
ã
	full_text~
|
z%34 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %22, i64 %26, i64 %27, i64 %29, i64 2
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %26
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
7fmul8B-
+
	full_text

%36 = fmul double %35, %35
+double8B

	full_text


double %35
+double8B

	full_text


double %35
icall8B_
]
	full_textP
N
L%37 = tail call double @llvm.fmuladd.f64(double %33, double %33, double %36)
+double8B

	full_text


double %33
+double8B

	full_text


double %33
+double8B

	full_text


double %36
¢getelementptr8Bé
ã
	full_text~
|
z%38 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %22, i64 %26, i64 %27, i64 %29, i64 3
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %26
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
icall8B_
]
	full_textP
N
L%40 = tail call double @llvm.fmuladd.f64(double %39, double %39, double %37)
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
¢getelementptr8Bé
ã
	full_text~
|
z%42 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %22, i64 %26, i64 %27, i64 %29, i64 0
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %26
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
7fdiv8B-
+
	full_text

%44 = fdiv double %41, %43
+double8B

	full_text


double %41
+double8B

	full_text


double %43
7fsub8B-
+
	full_text

%45 = fsub double %31, %44
+double8B

	full_text


double %31
+double8B

	full_text


double %44
@fmul8B6
4
	full_text'
%
#%46 = fmul double %45, 4.000000e-01
+double8B

	full_text


double %45
vgetelementptr8Bc
a
	full_textT
R
P%47 = getelementptr inbounds [35 x double], [35 x double]* %23, i64 %26, i64 %29
;[35 x double]*8B%
#
	full_text

[35 x double]* %23
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %29
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
4add8B+
)
	full_text

%48 = add nsw i32 %6, -1
6sext8B,
*
	full_text

%49 = sext i32 %48 to i64
%i328B

	full_text
	
i32 %48
¢getelementptr8Bé
ã
	full_text~
|
z%50 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %22, i64 %26, i64 %49, i64 %29, i64 4
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %49
%i648B

	full_text
	
i64 %29
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
z%52 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %22, i64 %26, i64 %49, i64 %29, i64 1
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %49
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
¢getelementptr8Bé
ã
	full_text~
|
z%54 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %22, i64 %26, i64 %49, i64 %29, i64 2
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %49
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%55 = load double, double* %54, align 8, !tbaa !8
-double*8B

	full_text

double* %54
7fmul8B-
+
	full_text

%56 = fmul double %55, %55
+double8B

	full_text


double %55
+double8B

	full_text


double %55
icall8B_
]
	full_textP
N
L%57 = tail call double @llvm.fmuladd.f64(double %53, double %53, double %56)
+double8B

	full_text


double %53
+double8B

	full_text


double %53
+double8B

	full_text


double %56
¢getelementptr8Bé
ã
	full_text~
|
z%58 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %22, i64 %26, i64 %49, i64 %29, i64 3
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %49
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
icall8B_
]
	full_textP
N
L%60 = tail call double @llvm.fmuladd.f64(double %59, double %59, double %57)
+double8B

	full_text


double %59
+double8B

	full_text


double %59
+double8B

	full_text


double %57
@fmul8B6
4
	full_text'
%
#%61 = fmul double %60, 5.000000e-01
+double8B

	full_text


double %60
¢getelementptr8Bé
ã
	full_text~
|
z%62 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %22, i64 %26, i64 %49, i64 %29, i64 0
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %49
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
7fdiv8B-
+
	full_text

%64 = fdiv double %61, %63
+double8B

	full_text


double %61
+double8B

	full_text


double %63
7fsub8B-
+
	full_text

%65 = fsub double %51, %64
+double8B

	full_text


double %51
+double8B

	full_text


double %64
@fmul8B6
4
	full_text'
%
#%66 = fmul double %65, 4.000000e-01
+double8B

	full_text


double %65
vgetelementptr8Bc
a
	full_textT
R
P%67 = getelementptr inbounds [35 x double], [35 x double]* %24, i64 %26, i64 %29
;[35 x double]*8B%
#
	full_text

[35 x double]* %24
%i648B

	full_text
	
i64 %26
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


double* %1
,double*8B

	full_text


double* %2
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %8
$i328B

	full_text


i32 %4
$i328B

	full_text


i32 %7
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
#i648B

	full_text	

i64 2
$i328B

	full_text


i32 -1
#i328B

	full_text	

i32 1
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
i64 4
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
4double8B&
$
	full_text

double 4.000000e-01        		 
 
 

                    !    "# "" $% $& $' $( $$ )* )) +, +- +. +/ ++ 01 00 23 24 25 26 22 78 77 9: 9; 99 <= <> <? << @A @B @C @D @@ EF EE GH GI GJ GG KL KK MN MO MP MQ MM RS RR TU TV TT WX WY WW Z[ ZZ \] \^ \_ \\ `a `b `` cc de dd fg fh fi fj ff kl kk mn mo mp mq mm rs rr tu tv tw tx tt yz yy {| {} {{ ~ ~	Ä ~	Å ~~ ÇÉ Ç
Ñ Ç
Ö Ç
Ü ÇÇ áà áá âä â
ã â
å ââ çé çç èê è
ë è
í è
ì èè îï îî ñó ñ
ò ññ ôö ô
õ ôô úù úú ûü û
† û
° ûû ¢£ ¢
§ ¢¢ •ß ® © 	™ 	´ ¨ ≠ Æ 	Ø c    	  
       
 !  # % & '" ($ * , - ." /+ 1 3 4 5" 62 87 :7 ;0 =0 >9 ? A B C" D@ FE HE I< JG L N O P" QM SK UR V) XT YW [ ] ^" _Z a\ bc e g hd i" jf l n od p" qm s u vd w" xt zy |y }r r Ä{ Å É Ñd Ö" ÜÇ àá äá ã~ åâ é ê ëd í" ìè ïç óî òk öñ õô ù ü †" °ú £û §  ¶• ¶ ¶ ∞∞ ±±< ±± < ∞∞  ∞∞ ~ ±± ~â ±± âG ±± G	≤ 2	≤ t	≥ c¥ 	µ 	µ 	µ  	µ "	∂ K
∂ ç∑ 	∏ @
∏ Ç	π $	π f	∫ +	∫ m	ª M
ª è	º Z
º ú"	
pintgr2"
_Z13get_global_idj"
llvm.fmuladd.f64*ä
npb-LU-pintgr2.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Ä

wgsize
>

wgsize_log1p
äùzA
 
transfer_bytes_log1p
äùzA

devmap_label
 

transfer_bytes
∞∞É