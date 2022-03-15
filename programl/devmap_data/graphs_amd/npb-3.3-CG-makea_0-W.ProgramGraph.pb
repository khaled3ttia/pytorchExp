

[external]
6allocaB,
*
	full_text

%10 = alloca i32, align 4
=allocaB3
1
	full_text$
"
 %11 = alloca [9 x i32], align 16
@allocaB6
4
	full_text'
%
#%12 = alloca [9 x double], align 16
9allocaB/
-
	full_text 

%13 = alloca double, align 8
;bitcastB0
.
	full_text!

%14 = bitcast i32* %10 to i8*
%i32*B

	full_text


i32* %10
YcallBQ
O
	full_textB
@
>call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %14) #4
#i8*B

	full_text
	
i8* %14
AbitcastB6
4
	full_text'
%
#%15 = bitcast [9 x i32]* %11 to i8*
1
[9 x i32]*B!

	full_text

[9 x i32]* %11
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 36, i8* nonnull %15) #4
#i8*B

	full_text
	
i8* %15
DbitcastB9
7
	full_text*
(
&%16 = bitcast [9 x double]* %12 to i8*
7[9 x double]*B$
"
	full_text

[9 x double]* %12
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 72, i8* nonnull %16) #4
#i8*B

	full_text
	
i8* %16
>bitcastB3
1
	full_text$
"
 %17 = bitcast double* %13 to i8*
+double*B

	full_text

double* %13
YcallBQ
O
	full_textB
@
>call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %17) #4
#i8*B

	full_text
	
i8* %17
KstoreBB
@
	full_text3
1
/store double %7, double* %13, align 8, !tbaa !8
+double*B

	full_text

double* %13
AbitcastB6
4
	full_text'
%
#%18 = bitcast i32* %1 to [9 x i32]*
GbitcastB<
:
	full_text-
+
)%19 = bitcast double* %2 to [9 x double]*
NcallBF
D
	full_text7
5
3%20 = tail call i64 @_Z15get_global_sizej(i32 0) #5
6truncB-
+
	full_text

%21 = trunc i64 %20 to i32
#i64B

	full_text
	
i64 %20
LcallBD
B
	full_text5
3
1%22 = tail call i64 @_Z13get_global_idj(i32 0) #5
6truncB-
+
	full_text

%23 = trunc i64 %22 to i32
#i64B

	full_text
	
i64 %22
.addB'
%
	full_text

%24 = add i32 %5, -1
0addB)
'
	full_text

%25 = add i32 %24, %21
#i32B

	full_text
	
i32 %24
#i32B

	full_text
	
i32 %21
2sdivB*
(
	full_text

%26 = sdiv i32 %25, %21
#i32B

	full_text
	
i32 %25
#i32B

	full_text
	
i32 %21
4mulB-
+
	full_text

%27 = mul nsw i32 %26, %23
#i32B

	full_text
	
i32 %26
#i32B

	full_text
	
i32 %23
4addB-
+
	full_text

%28 = add nsw i32 %27, %26
#i32B

	full_text
	
i32 %27
#i32B

	full_text
	
i32 %26
5icmpB-
+
	full_text

%29 = icmp sgt i32 %28, %5
#i32B

	full_text
	
i32 %28
AselectB7
5
	full_text(
&
$%30 = select i1 %29, i32 %5, i32 %28
!i1B

	full_text


i1 %29
#i32B

	full_text
	
i32 %28
/shlB(
&
	full_text

%31 = shl i64 %22, 32
#i64B

	full_text
	
i64 %22
7ashrB/
-
	full_text 

%32 = ashr exact i64 %31, 32
#i64B

	full_text
	
i64 %31
VgetelementptrBE
C
	full_text6
4
2%33 = getelementptr inbounds i32, i32* %3, i64 %32
#i64B

	full_text
	
i64 %32
GstoreB>
<
	full_text/
-
+store i32 %27, i32* %33, align 4, !tbaa !12
#i32B

	full_text
	
i32 %27
%i32*B

	full_text


i32* %33
VgetelementptrBE
C
	full_text6
4
2%34 = getelementptr inbounds i32, i32* %4, i64 %32
#i64B

	full_text
	
i64 %32
GstoreB>
<
	full_text/
-
+store i32 %30, i32* %34, align 4, !tbaa !12
#i32B

	full_text
	
i32 %30
%i32*B

	full_text


i32* %34
4icmpB,
*
	full_text

%35 = icmp sgt i32 %30, 0
#i32B

	full_text
	
i32 %30
8brB2
0
	full_text#
!
br i1 %35, label %36, label %96
!i1B

	full_text


i1 %35
pgetelementptr8B]
[
	full_textN
L
J%37 = getelementptr inbounds [9 x double], [9 x double]* %12, i64 0, i64 0
9[9 x double]*8B$
"
	full_text

[9 x double]* %12
jgetelementptr8BW
U
	full_textH
F
D%38 = getelementptr inbounds [9 x i32], [9 x i32]* %11, i64 0, i64 0
3
[9 x i32]*8B!

	full_text

[9 x i32]* %11
6sext8B,
*
	full_text

%39 = sext i32 %30 to i64
%i328B

	full_text
	
i32 %30
6sext8B,
*
	full_text

%40 = sext i32 %27 to i64
%i328B

	full_text
	
i32 %27
'br8B

	full_text

br label %41
Bphi8B9
7
	full_text*
(
&%42 = phi i64 [ 0, %36 ], [ %44, %94 ]
%i648B

	full_text
	
i64 %44
Gstore8B<
:
	full_text-
+
)store i32 8, i32* %10, align 4, !tbaa !12
'i32*8B

	full_text


i32* %10
‘call8B†
ƒ
	full_textv
t
rcall void @sprnvc(i32 %5, i32 8, i32 %6, double* nonnull %37, i32* nonnull %38, double* nonnull %13, double %8) #4
-double*8B

	full_text

double* %37
'i32*8B

	full_text


i32* %38
-double*8B

	full_text

double* %13
8icmp8B.
,
	full_text

%43 = icmp slt i64 %42, %40
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %40
8add8B/
-
	full_text 

%44 = add nuw nsw i64 %42, 1
%i648B

	full_text
	
i64 %42
:br8B2
0
	full_text#
!
br i1 %43, label %94, label %45
#i18B

	full_text


i1 %43
8trunc8B-
+
	full_text

%46 = trunc i64 %44 to i32
%i648B

	full_text
	
i64 %44
’call8B‡
„
	full_textw
u
scall void @vecset(i32 %5, double* nonnull %37, i32* nonnull %38, i32* nonnull %10, i32 %46, double 5.000000e-01) #4
-double*8B

	full_text

double* %37
'i32*8B

	full_text


i32* %38
'i32*8B

	full_text


i32* %10
%i328B

	full_text
	
i32 %46
Iload8B?
=
	full_text0
.
,%47 = load i32, i32* %10, align 4, !tbaa !12
'i32*8B

	full_text


i32* %10
Xgetelementptr8BE
C
	full_text6
4
2%48 = getelementptr inbounds i32, i32* %0, i64 %42
%i648B

	full_text
	
i64 %42
Istore8B>
<
	full_text/
-
+store i32 %47, i32* %48, align 4, !tbaa !12
%i328B

	full_text
	
i32 %47
'i32*8B

	full_text


i32* %48
6icmp8B,
*
	full_text

%49 = icmp sgt i32 %47, 0
%i328B

	full_text
	
i32 %47
:br8B2
0
	full_text#
!
br i1 %49, label %50, label %94
#i18B

	full_text


i1 %49
6zext8B,
*
	full_text

%51 = zext i32 %47 to i64
%i328B

	full_text
	
i32 %47
0and8B'
%
	full_text

%52 = and i64 %51, 1
%i648B

	full_text
	
i64 %51
5icmp8B+
)
	full_text

%53 = icmp eq i32 %47, 1
%i328B

	full_text
	
i32 %47
:br8B2
0
	full_text#
!
br i1 %53, label %81, label %54
#i18B

	full_text


i1 %53
6sub8B-
+
	full_text

%55 = sub nsw i64 %51, %52
%i648B

	full_text
	
i64 %51
%i648B

	full_text
	
i64 %52
'br8B

	full_text

br label %56
Bphi8B9
7
	full_text*
(
&%57 = phi i64 [ 0, %54 ], [ %78, %56 ]
%i648B

	full_text
	
i64 %78
Dphi8B;
9
	full_text,
*
(%58 = phi i64 [ %55, %54 ], [ %79, %56 ]
%i648B

	full_text
	
i64 %55
%i648B

	full_text
	
i64 %79
lgetelementptr8BY
W
	full_textJ
H
F%59 = getelementptr inbounds [9 x i32], [9 x i32]* %11, i64 0, i64 %57
3
[9 x i32]*8B!

	full_text

[9 x i32]* %11
%i648B

	full_text
	
i64 %57
Iload8B?
=
	full_text0
.
,%60 = load i32, i32* %59, align 8, !tbaa !12
'i32*8B

	full_text


i32* %59
5add8B,
*
	full_text

%61 = add nsw i32 %60, -1
%i328B

	full_text
	
i32 %60
ngetelementptr8B[
Y
	full_textL
J
H%62 = getelementptr inbounds [9 x i32], [9 x i32]* %18, i64 %42, i64 %57
3
[9 x i32]*8B!

	full_text

[9 x i32]* %18
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %57
Istore8B>
<
	full_text/
-
+store i32 %61, i32* %62, align 4, !tbaa !12
%i328B

	full_text
	
i32 %61
'i32*8B

	full_text


i32* %62
rgetelementptr8B_
]
	full_textP
N
L%63 = getelementptr inbounds [9 x double], [9 x double]* %12, i64 0, i64 %57
9[9 x double]*8B$
"
	full_text

[9 x double]* %12
%i648B

	full_text
	
i64 %57
Abitcast8B4
2
	full_text%
#
!%64 = bitcast double* %63 to i64*
-double*8B

	full_text

double* %63
Iload8B?
=
	full_text0
.
,%65 = load i64, i64* %64, align 16, !tbaa !8
'i64*8B

	full_text


i64* %64
tgetelementptr8Ba
_
	full_textR
P
N%66 = getelementptr inbounds [9 x double], [9 x double]* %19, i64 %42, i64 %57
9[9 x double]*8B$
"
	full_text

[9 x double]* %19
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %57
Abitcast8B4
2
	full_text%
#
!%67 = bitcast double* %66 to i64*
-double*8B

	full_text

double* %66
Hstore8B=
;
	full_text.
,
*store i64 %65, i64* %67, align 8, !tbaa !8
%i648B

	full_text
	
i64 %65
'i64*8B

	full_text


i64* %67
.or8B&
$
	full_text

%68 = or i64 %57, 1
%i648B

	full_text
	
i64 %57
lgetelementptr8BY
W
	full_textJ
H
F%69 = getelementptr inbounds [9 x i32], [9 x i32]* %11, i64 0, i64 %68
3
[9 x i32]*8B!

	full_text

[9 x i32]* %11
%i648B

	full_text
	
i64 %68
Iload8B?
=
	full_text0
.
,%70 = load i32, i32* %69, align 4, !tbaa !12
'i32*8B

	full_text


i32* %69
5add8B,
*
	full_text

%71 = add nsw i32 %70, -1
%i328B

	full_text
	
i32 %70
ngetelementptr8B[
Y
	full_textL
J
H%72 = getelementptr inbounds [9 x i32], [9 x i32]* %18, i64 %42, i64 %68
3
[9 x i32]*8B!

	full_text

[9 x i32]* %18
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %68
Istore8B>
<
	full_text/
-
+store i32 %71, i32* %72, align 4, !tbaa !12
%i328B

	full_text
	
i32 %71
'i32*8B

	full_text


i32* %72
rgetelementptr8B_
]
	full_textP
N
L%73 = getelementptr inbounds [9 x double], [9 x double]* %12, i64 0, i64 %68
9[9 x double]*8B$
"
	full_text

[9 x double]* %12
%i648B

	full_text
	
i64 %68
Abitcast8B4
2
	full_text%
#
!%74 = bitcast double* %73 to i64*
-double*8B

	full_text

double* %73
Hload8B>
<
	full_text/
-
+%75 = load i64, i64* %74, align 8, !tbaa !8
'i64*8B

	full_text


i64* %74
tgetelementptr8Ba
_
	full_textR
P
N%76 = getelementptr inbounds [9 x double], [9 x double]* %19, i64 %42, i64 %68
9[9 x double]*8B$
"
	full_text

[9 x double]* %19
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %68
Abitcast8B4
2
	full_text%
#
!%77 = bitcast double* %76 to i64*
-double*8B

	full_text

double* %76
Hstore8B=
;
	full_text.
,
*store i64 %75, i64* %77, align 8, !tbaa !8
%i648B

	full_text
	
i64 %75
'i64*8B

	full_text


i64* %77
4add8B+
)
	full_text

%78 = add nsw i64 %57, 2
%i648B

	full_text
	
i64 %57
1add8B(
&
	full_text

%79 = add i64 %58, -2
%i648B

	full_text
	
i64 %58
5icmp8B+
)
	full_text

%80 = icmp eq i64 %79, 0
%i648B

	full_text
	
i64 %79
:br8B2
0
	full_text#
!
br i1 %80, label %81, label %56
#i18B

	full_text


i1 %80
Bphi8B9
7
	full_text*
(
&%82 = phi i64 [ 0, %50 ], [ %78, %56 ]
%i648B

	full_text
	
i64 %78
5icmp8B+
)
	full_text

%83 = icmp eq i64 %52, 0
%i648B

	full_text
	
i64 %52
:br8B2
0
	full_text#
!
br i1 %83, label %94, label %84
#i18B

	full_text


i1 %83
lgetelementptr8BY
W
	full_textJ
H
F%85 = getelementptr inbounds [9 x i32], [9 x i32]* %11, i64 0, i64 %82
3
[9 x i32]*8B!

	full_text

[9 x i32]* %11
%i648B

	full_text
	
i64 %82
Iload8B?
=
	full_text0
.
,%86 = load i32, i32* %85, align 4, !tbaa !12
'i32*8B

	full_text


i32* %85
5add8B,
*
	full_text

%87 = add nsw i32 %86, -1
%i328B

	full_text
	
i32 %86
ngetelementptr8B[
Y
	full_textL
J
H%88 = getelementptr inbounds [9 x i32], [9 x i32]* %18, i64 %42, i64 %82
3
[9 x i32]*8B!

	full_text

[9 x i32]* %18
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %82
Istore8B>
<
	full_text/
-
+store i32 %87, i32* %88, align 4, !tbaa !12
%i328B

	full_text
	
i32 %87
'i32*8B

	full_text


i32* %88
rgetelementptr8B_
]
	full_textP
N
L%89 = getelementptr inbounds [9 x double], [9 x double]* %12, i64 0, i64 %82
9[9 x double]*8B$
"
	full_text

[9 x double]* %12
%i648B

	full_text
	
i64 %82
Abitcast8B4
2
	full_text%
#
!%90 = bitcast double* %89 to i64*
-double*8B

	full_text

double* %89
Hload8B>
<
	full_text/
-
+%91 = load i64, i64* %90, align 8, !tbaa !8
'i64*8B

	full_text


i64* %90
tgetelementptr8Ba
_
	full_textR
P
N%92 = getelementptr inbounds [9 x double], [9 x double]* %19, i64 %42, i64 %82
9[9 x double]*8B$
"
	full_text

[9 x double]* %19
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %82
Abitcast8B4
2
	full_text%
#
!%93 = bitcast double* %92 to i64*
-double*8B

	full_text

double* %92
Hstore8B=
;
	full_text.
,
*store i64 %91, i64* %93, align 8, !tbaa !8
%i648B

	full_text
	
i64 %91
'i64*8B

	full_text


i64* %93
'br8B

	full_text

br label %94
8icmp8	B.
,
	full_text

%95 = icmp slt i64 %44, %39
%i648	B

	full_text
	
i64 %44
%i648	B

	full_text
	
i64 %39
:br8	B2
0
	full_text#
!
br i1 %95, label %41, label %96
#i18	B

	full_text


i1 %95
Ycall8
BO
M
	full_text@
>
<call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %17) #4
%i8*8
B

	full_text
	
i8* %17
Zcall8
BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 72, i8* nonnull %16) #4
%i8*8
B

	full_text
	
i8* %16
Zcall8
BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 36, i8* nonnull %15) #4
%i8*8
B

	full_text
	
i8* %15
Ycall8
BO
M
	full_text@
>
<call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %14) #4
%i8*8
B

	full_text
	
i8* %14
$ret8
B

	full_text


ret void
,double*8B

	full_text


double* %2
*double8B

	full_text

	double %7
&i32*8B

	full_text
	
i32* %3
&i32*8B

	full_text
	
i32* %0
*double8B

	full_text

	double %8
&i32*8B

	full_text
	
i32* %4
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %6
&i32*8B

	full_text
	
i32* %1
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
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 72
$i648B

	full_text


i64 32
$i648B

	full_text


i64 36
#i648B

	full_text	

i64 0
4double8B&
$
	full_text

double 5.000000e-01
#i648B

	full_text	

i64 4
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 2
$i648B

	full_text


i64 -2
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 8
#i328B

	full_text	

i32 8        	
 		                       !  "    #$ #% ## &' &( && )* )+ )) ,- ,, ./ .0 .. 12 11 34 33 56 55 78 79 77 :; :: <= <> << ?@ ?? AB AD CC EF EE GH GG IJ II KM LL NO NN PQ PR PS PP TU TV TT WX WW YZ Y\ [[ ]^ ]_ ]` ]a ]] bc bb de dd fg fh ff ij ii kl kn mm op oo qr qq st sv uw uu xz yy {| {} {{ ~ ~	€ ~~ ‚  ƒ„ ƒƒ …† …
‡ …
 …… ‰ ‰
‹ ‰‰  
    ‘’ ‘‘ “” “
• “
– ““ — —— ™ ™
› ™™    
   ΅Ά ΅΅ £¤ ££ ¥¦ ¥
§ ¥
¨ ¥¥ ©ª ©
« ©© ¬­ ¬
® ¬¬ ―° ―― ±² ±± ³΄ ³
µ ³
¶ ³³ ·Έ ·· ΉΊ Ή
» ΉΉ Ό½ ΌΌ ΎΏ ΎΎ ΐΑ ΐΐ ΒΓ Β
Ε ΔΔ ΖΗ ΖΖ ΘΙ ΘΛ Κ
Μ ΚΚ ΝΞ ΝΝ ΟΠ ΟΟ ΡÒ Ρ
Σ Ρ
Τ ΡΡ ΥΦ Υ
Χ ΥΥ ΨΩ Ψ
Ϊ ΨΨ Ϋά ΫΫ έή έέ ίΰ ί
α ί
β ίί γδ γγ εζ ε
η εε θκ ι
λ ιι μν μ
ο ξξ π
ρ ππ ς
σ ςς τ
υ ττ φχ ψ ω 5ϊ d	ϋ Pό :ύ 	ύ ,	ύ .ύ Pύ ]	ώ Pÿ    
	         ! "  $ %# ' (& *# +) -, /) 0 21 43 6& 85 93 ;. =: >. @? B D F. H& JW M OC QE R SL UI VL XT ZW \C ^E _ `[ a cL eb gd hb ji lb nm pb rq tm vo wΌ zu |Ύ } y €~ ‚ „ †L ‡y ƒ … ‹ y   ’ ”L •y –“ ‘ — ›y     Ά΅ ¤ ¦L § ¨£ ª¥ « ­ ®¬ °― ² ΄L µ ¶³ Έ± Ί· »y ½{ ΏΎ Αΐ ΓΌ Εo ΗΖ Ι ΛΔ ΜΚ ΞΝ Π ÒL ΣΔ ΤΟ ΦΡ Χ ΩΔ ΪΨ άΫ ή ΰL αΔ βί δέ ζγ ηW κG λι ν ο ρ	 σ υA CA ξK LY ιY [μ Lμ ξk mk ιs Δs uΘ ιΘ Κx yθ ιΒ ΔΒ y  ƒƒ φ €€ „„ …… ‚‚ €€ ξ …… ξπ …… π €€   ς …… ςP ƒƒ P €€ ] „„ ]τ …… τ €€  ‚‚ † † 	† ?	† i‡ ‡ π	 1	 3‰ ‰ ς	 C	 C	 E	 E L y	 ~
 
 
 ¬
 ΐ Δ
 Ζ
 Κ
 Ψ	‹ ]  τ	 
 ƒ
 £
 Ο	 W	 o
 
 Ό
 Ύ‘ ‘ ‘ ‘ 	‘ q’ ’ ξ“ N	“ P"	
makea_0"
llvm.lifetime.start.p0i8"
_Z15get_global_sizej"
_Z13get_global_idj"
sprnvc"
vecset"
llvm.lifetime.end.p0i8*
npb-CG-makea_0.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02€

wgsize_log1p
½„A

devmap_label

 
transfer_bytes_log1p
½„A

wgsize
 

transfer_bytes
όδƒ