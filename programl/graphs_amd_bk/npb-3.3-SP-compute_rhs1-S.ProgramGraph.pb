

[external]
LcallBD
B
	full_text5
3
1%12 = tail call i64 @_Z13get_global_idj(i32 2) #3
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
LcallBD
B
	full_text5
3
1%15 = tail call i64 @_Z13get_global_idj(i32 0) #3
6truncB-
+
	full_text

%16 = trunc i64 %15 to i32
#i64B

	full_text
	
i64 %15
6icmpB.
,
	full_text

%17 = icmp slt i32 %13, %10
#i32B

	full_text
	
i32 %13
6truncB-
+
	full_text

%18 = trunc i64 %14 to i32
#i64B

	full_text
	
i64 %14
5icmpB-
+
	full_text

%19 = icmp slt i32 %18, %9
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
5icmpB-
+
	full_text

%21 = icmp slt i32 %16, %8
#i32B

	full_text
	
i32 %16
/andB(
&
	full_text

%22 = and i1 %20, %21
!i1B

	full_text


i1 %20
!i1B

	full_text


i1 %21
8brB2
0
	full_text#
!
br i1 %22, label %23, label %69
!i1B

	full_text


i1 %22
Wbitcast8BJ
H
	full_text;
9
7%24 = bitcast double* %0 to [13 x [13 x [5 x double]]]*
Qbitcast8BD
B
	full_text5
3
1%25 = bitcast double* %1 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%26 = bitcast double* %2 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%27 = bitcast double* %3 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%28 = bitcast double* %4 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%29 = bitcast double* %5 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%30 = bitcast double* %6 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%31 = bitcast double* %7 to [13 x [13 x double]]*
1shl8B(
&
	full_text

%32 = shl i64 %12, 32
%i648B

	full_text
	
i64 %12
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
1shl8B(
&
	full_text

%36 = shl i64 %15, 32
%i648B

	full_text
	
i64 %15
9ashr8B/
-
	full_text 

%37 = ashr exact i64 %36, 32
%i648B

	full_text
	
i64 %36
¢getelementptr8Bé
ã
	full_text~
|
z%38 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %24, i64 %33, i64 %35, i64 %37, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %24
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %37
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
z%40 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %24, i64 %33, i64 %35, i64 %37, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %24
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %37
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
z%42 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %24, i64 %33, i64 %35, i64 %37, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %24
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %37
Nload8BD
B
	full_text5
3
1%43 = load double, double* %42, align 8, !tbaa !8
-double*8B

	full_text

double* %42
¢getelementptr8Bé
ã
	full_text~
|
z%44 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %24, i64 %33, i64 %35, i64 %37, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %24
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %37
Nload8BD
B
	full_text5
3
1%45 = load double, double* %44, align 8, !tbaa !8
-double*8B

	full_text

double* %44
¢getelementptr8Bé
ã
	full_text~
|
z%46 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %24, i64 %33, i64 %35, i64 %37, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %24
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %37
Nload8BD
B
	full_text5
3
1%47 = load double, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
@fdiv8B6
4
	full_text'
%
#%48 = fdiv double 1.000000e+00, %39
+double8B

	full_text


double %39
çgetelementptr8Bz
x
	full_textk
i
g%49 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %29, i64 %33, i64 %35, i64 %37
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %29
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %37
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

%50 = fmul double %41, %48
+double8B

	full_text


double %41
+double8B

	full_text


double %48
çgetelementptr8Bz
x
	full_textk
i
g%51 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %25, i64 %33, i64 %35, i64 %37
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %25
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %37
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

%52 = fmul double %48, %43
+double8B

	full_text


double %48
+double8B

	full_text


double %43
çgetelementptr8Bz
x
	full_textk
i
g%53 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %26, i64 %33, i64 %35, i64 %37
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %26
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %37
Nstore8BC
A
	full_text4
2
0store double %52, double* %53, align 8, !tbaa !8
+double8B

	full_text


double %52
-double*8B

	full_text

double* %53
7fmul8B-
+
	full_text

%54 = fmul double %48, %45
+double8B

	full_text


double %48
+double8B

	full_text


double %45
çgetelementptr8Bz
x
	full_textk
i
g%55 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %27, i64 %33, i64 %35, i64 %37
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %27
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %37
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
7fmul8B-
+
	full_text

%56 = fmul double %43, %43
+double8B

	full_text


double %43
+double8B

	full_text


double %43
icall8B_
]
	full_textP
N
L%57 = tail call double @llvm.fmuladd.f64(double %41, double %41, double %56)
+double8B

	full_text


double %41
+double8B

	full_text


double %41
+double8B

	full_text


double %56
icall8B_
]
	full_textP
N
L%58 = tail call double @llvm.fmuladd.f64(double %45, double %45, double %57)
+double8B

	full_text


double %45
+double8B

	full_text


double %45
+double8B

	full_text


double %57
@fmul8B6
4
	full_text'
%
#%59 = fmul double %58, 5.000000e-01
+double8B

	full_text


double %58
7fmul8B-
+
	full_text

%60 = fmul double %48, %59
+double8B

	full_text


double %48
+double8B

	full_text


double %59
çgetelementptr8Bz
x
	full_textk
i
g%61 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %31, i64 %33, i64 %35, i64 %37
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %37
Nstore8BC
A
	full_text4
2
0store double %60, double* %61, align 8, !tbaa !8
+double8B

	full_text


double %60
-double*8B

	full_text

double* %61
7fmul8B-
+
	full_text

%62 = fmul double %48, %60
+double8B

	full_text


double %48
+double8B

	full_text


double %60
çgetelementptr8Bz
x
	full_textk
i
g%63 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %28, i64 %33, i64 %35, i64 %37
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %28
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %37
Nstore8BC
A
	full_text4
2
0store double %62, double* %63, align 8, !tbaa !8
+double8B

	full_text


double %62
-double*8B

	full_text

double* %63
Ffmul8B<
:
	full_text-
+
)%64 = fmul double %48, 0x3FE1EB851EB851EB
+double8B

	full_text


double %48
7fsub8B-
+
	full_text

%65 = fsub double %47, %60
+double8B

	full_text


double %47
+double8B

	full_text


double %60
7fmul8B-
+
	full_text

%66 = fmul double %64, %65
+double8B

	full_text


double %64
+double8B

	full_text


double %65
Lcall8BB
@
	full_text3
1
/%67 = tail call double @_Z4sqrtd(double %66) #3
+double8B

	full_text


double %66
çgetelementptr8Bz
x
	full_textk
i
g%68 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %30, i64 %33, i64 %35, i64 %37
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %30
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %37
Nstore8BC
A
	full_text4
2
0store double %67, double* %68, align 8, !tbaa !8
+double8B

	full_text


double %67
-double*8B

	full_text

double* %68
'br8B

	full_text

br label %69
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


double* %5
,double*8B

	full_text


double* %6
,double*8B

	full_text


double* %3
,double*8B

	full_text


double* %1
%i328B

	full_text
	
i32 %10
,double*8B

	full_text


double* %4
,double*8B

	full_text


double* %7
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %9
$i328B

	full_text


i32 %8
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
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 2
:double8B,
*
	full_text

double 0x3FE1EB851EB851EB
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 4
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
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 0
4double8B&
$
	full_text

double 1.000000e+00       	  
 

                     !    "# "" $% $$ &' && () (( *+ ** ,- ,. ,/ ,0 ,, 12 11 34 35 36 37 33 89 88 :; :< := :> :: ?@ ?? AB AC AD AE AA FG FF HI HJ HK HL HH MN MM OP OO QR QS QT QU QQ VW VX VV YZ Y[ YY \] \^ \_ \` \\ ab ac aa de df dd gh gi gj gk gg lm ln ll op oq oo rs rt ru rv rr wx wy ww z{ z| zz }~ } }	Ä }} ÅÇ Å
É Å
Ñ ÅÅ ÖÜ ÖÖ áà á
â áá äã ä
å ä
ç ä
é ää èê è
ë èè íì í
î íí ïñ ï
ó ï
ò ï
ô ïï öõ ö
ú öö ùû ùù ü† ü
° üü ¢£ ¢
§ ¢¢ •¶ •• ß® ß
© ß
™ ß
´ ßß ¨≠ ¨
Æ ¨¨ Ø± ≤ ≥ ¥ µ 	∂ ∑ ∏ π 	∫ 	ª    	 
        !  # %$ ' )( + -" .& /* 0, 2 4" 5& 6* 73 9 ;" <& =* >: @ B" C& D* EA G I" J& K* LH N1 P R" S& T* UO WQ X8 ZO [ ]" ^& _* `Y b\ cO e? f h" i& j* kd mg nO pF q s" t& u* vo xr y? {? |8 ~8 z ÄF ÇF É} ÑÅ ÜO àÖ â ã" å& ç* éá êä ëO ìá î ñ" ó& ò* ôí õï úO ûM †á °ù £ü §¢ ¶ ®" ©& ™* ´• ≠ß Æ  ∞Ø ∞ ææ ∞ ΩΩ ºº ºº  ºº } ΩΩ }• ææ •Å ΩΩ Å ºº ø 	¿ :
¡ ù	¬  	¬ "	¬ $	¬ &	¬ (	¬ *	√ H	ƒ A
≈ Ö∆ 	« ,	» 3…   O"
compute_rhs1"
_Z13get_global_idj"
llvm.fmuladd.f64"

_Z4sqrtd*è
npb-SP-compute_rhs1.clu
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

transfer_bytes
∏ä1

wgsize
<

devmap_label
 