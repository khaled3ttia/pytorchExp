

[external]
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 0) #3
4truncB+
)
	full_text

%9 = trunc i64 %8 to i32
"i64B

	full_text


i64 %8
4icmpB,
*
	full_text

%10 = icmp slt i32 %9, %5
"i32B

	full_text


i32 %9
8brB2
0
	full_text#
!
br i1 %10, label %11, label %67
!i1B

	full_text


i1 %10
0shl8B'
%
	full_text

%12 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%13 = ashr exact i64 %12, 32
%i648B

	full_text
	
i64 %12
\getelementptr8BI
G
	full_text:
8
6%14 = getelementptr inbounds float, float* %1, i64 %13
%i648B

	full_text
	
i64 %13
Ustore8BJ
H
	full_text;
9
7store float 0.000000e+00, float* %14, align 4, !tbaa !9
+float*8B

	full_text


float* %14
5icmp8B+
)
	full_text

%15 = icmp sgt i32 %6, 0
:br8B2
0
	full_text#
!
br i1 %15, label %16, label %61
#i18B

	full_text


i1 %15
\getelementptr8BI
G
	full_text:
8
6%17 = getelementptr inbounds float, float* %0, i64 %13
%i648B

	full_text
	
i64 %13
5sext8B+
)
	full_text

%18 = sext i32 %5 to i64
0shl8B'
%
	full_text

%19 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%20 = ashr exact i64 %19, 32
%i648B

	full_text
	
i64 %19
5zext8B+
)
	full_text

%21 = zext i32 %6 to i64
0and8B'
%
	full_text

%22 = and i64 %21, 1
%i648B

	full_text
	
i64 %21
4icmp8B*
(
	full_text

%23 = icmp eq i32 %6, 1
:br8B2
0
	full_text#
!
br i1 %23, label %48, label %24
#i18B

	full_text


i1 %23
6sub8B-
+
	full_text

%25 = sub nsw i64 %21, %22
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %22
'br8B

	full_text

br label %26
Ophi8BF
D
	full_text7
5
3%27 = phi float [ 0.000000e+00, %24 ], [ %44, %26 ]
)float8B

	full_text

	float %44
Bphi8B9
7
	full_text*
(
&%28 = phi i64 [ 0, %24 ], [ %45, %26 ]
%i648B

	full_text
	
i64 %45
Dphi8B;
9
	full_text,
*
(%29 = phi i64 [ %25, %24 ], [ %46, %26 ]
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %46
6mul8B-
+
	full_text

%30 = mul nsw i64 %28, %18
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %18
6add8B-
+
	full_text

%31 = add nsw i64 %30, %20
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %20
\getelementptr8BI
G
	full_text:
8
6%32 = getelementptr inbounds float, float* %2, i64 %31
%i648B

	full_text
	
i64 %31
Lload8BB
@
	full_text3
1
/%33 = load float, float* %32, align 4, !tbaa !9
+float*8B

	full_text


float* %32
Lload8BB
@
	full_text3
1
/%34 = load float, float* %17, align 4, !tbaa !9
+float*8B

	full_text


float* %17
6fsub8B,
*
	full_text

%35 = fsub float %33, %34
)float8B

	full_text

	float %33
)float8B

	full_text

	float %34
ecall8B[
Y
	full_textL
J
H%36 = tail call float @llvm.fmuladd.f32(float %35, float %35, float %27)
)float8B

	full_text

	float %35
)float8B

	full_text

	float %35
)float8B

	full_text

	float %27
Lstore8BA
?
	full_text2
0
.store float %36, float* %14, align 4, !tbaa !9
)float8B

	full_text

	float %36
+float*8B

	full_text


float* %14
.or8B&
$
	full_text

%37 = or i64 %28, 1
%i648B

	full_text
	
i64 %28
6mul8B-
+
	full_text

%38 = mul nsw i64 %37, %18
%i648B

	full_text
	
i64 %37
%i648B

	full_text
	
i64 %18
6add8B-
+
	full_text

%39 = add nsw i64 %38, %20
%i648B

	full_text
	
i64 %38
%i648B

	full_text
	
i64 %20
\getelementptr8BI
G
	full_text:
8
6%40 = getelementptr inbounds float, float* %2, i64 %39
%i648B

	full_text
	
i64 %39
Lload8BB
@
	full_text3
1
/%41 = load float, float* %40, align 4, !tbaa !9
+float*8B

	full_text


float* %40
Lload8BB
@
	full_text3
1
/%42 = load float, float* %17, align 4, !tbaa !9
+float*8B

	full_text


float* %17
6fsub8B,
*
	full_text

%43 = fsub float %41, %42
)float8B

	full_text

	float %41
)float8B

	full_text

	float %42
ecall8B[
Y
	full_textL
J
H%44 = tail call float @llvm.fmuladd.f32(float %43, float %43, float %36)
)float8B

	full_text

	float %43
)float8B

	full_text

	float %43
)float8B

	full_text

	float %36
Lstore8BA
?
	full_text2
0
.store float %44, float* %14, align 4, !tbaa !9
)float8B

	full_text

	float %44
+float*8B

	full_text


float* %14
4add8B+
)
	full_text

%45 = add nsw i64 %28, 2
%i648B

	full_text
	
i64 %28
1add8B(
&
	full_text

%46 = add i64 %29, -2
%i648B

	full_text
	
i64 %29
5icmp8B+
)
	full_text

%47 = icmp eq i64 %46, 0
%i648B

	full_text
	
i64 %46
:br8B2
0
	full_text#
!
br i1 %47, label %48, label %26
#i18B

	full_text


i1 %47
Hphi8B?
=
	full_text0
.
,%49 = phi float [ undef, %16 ], [ %44, %26 ]
)float8B

	full_text

	float %44
Ophi8BF
D
	full_text7
5
3%50 = phi float [ 0.000000e+00, %16 ], [ %44, %26 ]
)float8B

	full_text

	float %44
Bphi8B9
7
	full_text*
(
&%51 = phi i64 [ 0, %16 ], [ %45, %26 ]
%i648B

	full_text
	
i64 %45
5icmp8B+
)
	full_text

%52 = icmp eq i64 %22, 0
%i648B

	full_text
	
i64 %22
:br8B2
0
	full_text#
!
br i1 %52, label %61, label %53
#i18B

	full_text


i1 %52
6mul8B-
+
	full_text

%54 = mul nsw i64 %51, %18
%i648B

	full_text
	
i64 %51
%i648B

	full_text
	
i64 %18
6add8B-
+
	full_text

%55 = add nsw i64 %54, %20
%i648B

	full_text
	
i64 %54
%i648B

	full_text
	
i64 %20
\getelementptr8BI
G
	full_text:
8
6%56 = getelementptr inbounds float, float* %2, i64 %55
%i648B

	full_text
	
i64 %55
Lload8BB
@
	full_text3
1
/%57 = load float, float* %56, align 4, !tbaa !9
+float*8B

	full_text


float* %56
Lload8BB
@
	full_text3
1
/%58 = load float, float* %17, align 4, !tbaa !9
+float*8B

	full_text


float* %17
6fsub8B,
*
	full_text

%59 = fsub float %57, %58
)float8B

	full_text

	float %57
)float8B

	full_text

	float %58
ecall8B[
Y
	full_textL
J
H%60 = tail call float @llvm.fmuladd.f32(float %59, float %59, float %50)
)float8B

	full_text

	float %59
)float8B

	full_text

	float %59
)float8B

	full_text

	float %50
Lstore8BA
?
	full_text2
0
.store float %60, float* %14, align 4, !tbaa !9
)float8B

	full_text

	float %60
+float*8B

	full_text


float* %14
'br8B

	full_text

br label %61
]phi8BT
R
	full_textE
C
A%62 = phi float [ 0.000000e+00, %11 ], [ %49, %48 ], [ %60, %53 ]
)float8B

	full_text

	float %49
)float8B

	full_text

	float %60
Bfdiv8B8
6
	full_text)
'
%%63 = fdiv float %62, %3, !fpmath !13
)float8B

	full_text

	float %62
Jcall8B@
>
	full_text1
/
-%64 = tail call float @_Z4sqrtf(float %63) #3
)float8B

	full_text

	float %63
9fcmp8B/
-
	full_text 

%65 = fcmp ugt float %64, %4
)float8B

	full_text

	float %64
Qselect8BE
C
	full_text6
4
2%66 = select i1 %65, float %64, float 1.000000e+00
#i18B

	full_text


i1 %65
)float8B

	full_text

	float %64
Lstore8BA
?
	full_text2
0
.store float %66, float* %14, align 4, !tbaa !9
)float8B

	full_text

	float %66
+float*8B

	full_text


float* %14
'br8B

	full_text

br label %67
$ret8B

	full_text


ret void
*float*8	B

	full_text

	float* %0
*float*8	B

	full_text

	float* %2
$i328	B

	full_text


i32 %6
$i328	B

	full_text


i32 %5
(float8	B

	full_text


float %4
(float8	B

	full_text


float %3
*float*8	B

	full_text

	float* %1
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
+float8	B

	full_text

float undef
#i648	B

	full_text	

i64 1
#i328	B

	full_text	

i32 1
#i328	B

	full_text	

i32 0
#i648	B

	full_text	

i64 2
2float8	B%
#
	full_text

float 0.000000e+00
2float8	B%
#
	full_text

float 1.000000e+00
$i648	B

	full_text


i64 32
$i648	B

	full_text


i64 -2
#i648	B

	full_text	

i64 0      	  
 

                   !  "    #% $$ &' && () (* (( +, +- ++ ./ .0 .. 12 11 34 33 56 55 78 79 77 :; :< := :: >? >@ >> AB AA CD CE CC FG FH FF IJ II KL KK MN MM OP OQ OO RS RT RU RR VW VX VV YZ YY [\ [[ ]^ ]] _` _b aa cd cc ef ee gh gg ij il km kk no np nn qr qq st ss uv uu wx wy ww z{ z| z} zz ~ ~	Ä ~~ Å
É Ç
Ñ ÇÇ ÖÜ ÖÖ áà áá âä ââ ãå ã
ç ãã éè é
ê éé ëì î 1î Iî qï ï ï 	ñ ñ 
ó â
ò Öô     	 
   
      ! "R %Y '  )[ *& , -+ / 0. 21 4 63 85 97 ;7 <$ =: ? @& BA D EC G HF JI L NK PM QO SO T: UR W X& Z( \[ ^] `R bR dY f hg je l mk o pn rq t vs xu yw {w |c }z  Äa Éz ÑÇ ÜÖ àá äâ åá çã è ê  í  Ç a  ë íi Çi k# $Å Ç_ a_ $ í õõ úú ööR õõ Rz õõ zá úú á: õõ : öö ù a	û 	û A	ü † 	† 	° Y¢ ¢ $¢ c¢ Ç
£ ã	§ 	§ 
	§ 	§ 	• [¶ &	¶ ]¶ e	¶ g"

std_kernel"
_Z13get_global_idj"
llvm.fmuladd.f32"

_Z4sqrtf*§
+polybench-gpu-1.0-correlation-std_kernel.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å
 
transfer_bytes_log1p
"§äA

transfer_bytes
êÄÉ

devmap_label


wgsize_log1p
"§äA

wgsize
Ä