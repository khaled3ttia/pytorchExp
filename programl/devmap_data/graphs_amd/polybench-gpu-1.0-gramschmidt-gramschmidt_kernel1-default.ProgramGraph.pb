

[external]
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 0) #3
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
1icmpB)
'
	full_text

%9 = icmp eq i32 %8, 0
"i32B

	full_text


i32 %8
7brB1
/
	full_text"
 
br i1 %9, label %10, label %76
 i1B

	full_text	

i1 %9
5icmp8B+
)
	full_text

%11 = icmp sgt i32 %4, 0
:br8B2
0
	full_text#
!
br i1 %11, label %12, label %69
#i18B

	full_text


i1 %11
5sext8B+
)
	full_text

%13 = sext i32 %5 to i64
5sext8B+
)
	full_text

%14 = sext i32 %3 to i64
5zext8B+
)
	full_text

%15 = zext i32 %4 to i64
5add8B,
*
	full_text

%16 = add nsw i64 %15, -1
%i648B

	full_text
	
i64 %15
0and8B'
%
	full_text

%17 = and i64 %15, 3
%i648B

	full_text
	
i64 %15
6icmp8B,
*
	full_text

%18 = icmp ult i64 %16, 3
%i648B

	full_text
	
i64 %16
:br8B2
0
	full_text#
!
br i1 %18, label %51, label %19
#i18B

	full_text


i1 %18
6sub8B-
+
	full_text

%20 = sub nsw i64 %15, %17
%i648B

	full_text
	
i64 %15
%i648B

	full_text
	
i64 %17
'br8B

	full_text

br label %21
Bphi8B9
7
	full_text*
(
&%22 = phi i64 [ 0, %19 ], [ %48, %21 ]
%i648B

	full_text
	
i64 %48
Ophi8BF
D
	full_text7
5
3%23 = phi float [ 0.000000e+00, %19 ], [ %47, %21 ]
)float8B

	full_text

	float %47
Dphi8B;
9
	full_text,
*
(%24 = phi i64 [ %20, %19 ], [ %49, %21 ]
%i648B

	full_text
	
i64 %20
%i648B

	full_text
	
i64 %49
6mul8B-
+
	full_text

%25 = mul nsw i64 %22, %13
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %13
6add8B-
+
	full_text

%26 = add nsw i64 %25, %14
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %14
\getelementptr8BI
G
	full_text:
8
6%27 = getelementptr inbounds float, float* %0, i64 %26
%i648B

	full_text
	
i64 %26
Lload8BB
@
	full_text3
1
/%28 = load float, float* %27, align 4, !tbaa !9
+float*8B

	full_text


float* %27
ecall8B[
Y
	full_textL
J
H%29 = tail call float @llvm.fmuladd.f32(float %28, float %28, float %23)
)float8B

	full_text

	float %28
)float8B

	full_text

	float %28
)float8B

	full_text

	float %23
.or8B&
$
	full_text

%30 = or i64 %22, 1
%i648B

	full_text
	
i64 %22
6mul8B-
+
	full_text

%31 = mul nsw i64 %30, %13
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %13
6add8B-
+
	full_text

%32 = add nsw i64 %31, %14
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %14
\getelementptr8BI
G
	full_text:
8
6%33 = getelementptr inbounds float, float* %0, i64 %32
%i648B

	full_text
	
i64 %32
Lload8BB
@
	full_text3
1
/%34 = load float, float* %33, align 4, !tbaa !9
+float*8B

	full_text


float* %33
ecall8B[
Y
	full_textL
J
H%35 = tail call float @llvm.fmuladd.f32(float %34, float %34, float %29)
)float8B

	full_text

	float %34
)float8B

	full_text

	float %34
)float8B

	full_text

	float %29
.or8B&
$
	full_text

%36 = or i64 %22, 2
%i648B

	full_text
	
i64 %22
6mul8B-
+
	full_text

%37 = mul nsw i64 %36, %13
%i648B

	full_text
	
i64 %36
%i648B

	full_text
	
i64 %13
6add8B-
+
	full_text

%38 = add nsw i64 %37, %14
%i648B

	full_text
	
i64 %37
%i648B

	full_text
	
i64 %14
\getelementptr8BI
G
	full_text:
8
6%39 = getelementptr inbounds float, float* %0, i64 %38
%i648B

	full_text
	
i64 %38
Lload8BB
@
	full_text3
1
/%40 = load float, float* %39, align 4, !tbaa !9
+float*8B

	full_text


float* %39
ecall8B[
Y
	full_textL
J
H%41 = tail call float @llvm.fmuladd.f32(float %40, float %40, float %35)
)float8B

	full_text

	float %40
)float8B

	full_text

	float %40
)float8B

	full_text

	float %35
.or8B&
$
	full_text

%42 = or i64 %22, 3
%i648B

	full_text
	
i64 %22
6mul8B-
+
	full_text

%43 = mul nsw i64 %42, %13
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %13
6add8B-
+
	full_text

%44 = add nsw i64 %43, %14
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %14
\getelementptr8BI
G
	full_text:
8
6%45 = getelementptr inbounds float, float* %0, i64 %44
%i648B

	full_text
	
i64 %44
Lload8BB
@
	full_text3
1
/%46 = load float, float* %45, align 4, !tbaa !9
+float*8B

	full_text


float* %45
ecall8B[
Y
	full_textL
J
H%47 = tail call float @llvm.fmuladd.f32(float %46, float %46, float %41)
)float8B

	full_text

	float %46
)float8B

	full_text

	float %46
)float8B

	full_text

	float %41
4add8B+
)
	full_text

%48 = add nsw i64 %22, 4
%i648B

	full_text
	
i64 %22
1add8B(
&
	full_text

%49 = add i64 %24, -4
%i648B

	full_text
	
i64 %24
5icmp8B+
)
	full_text

%50 = icmp eq i64 %49, 0
%i648B

	full_text
	
i64 %49
:br8B2
0
	full_text#
!
br i1 %50, label %51, label %21
#i18B

	full_text


i1 %50
Hphi8B?
=
	full_text0
.
,%52 = phi float [ undef, %12 ], [ %47, %21 ]
)float8B

	full_text

	float %47
Bphi8B9
7
	full_text*
(
&%53 = phi i64 [ 0, %12 ], [ %48, %21 ]
%i648B

	full_text
	
i64 %48
Ophi8BF
D
	full_text7
5
3%54 = phi float [ 0.000000e+00, %12 ], [ %47, %21 ]
)float8B

	full_text

	float %47
5icmp8B+
)
	full_text

%55 = icmp eq i64 %17, 0
%i648B

	full_text
	
i64 %17
:br8B2
0
	full_text#
!
br i1 %55, label %69, label %56
#i18B

	full_text


i1 %55
'br8B

	full_text

br label %57
Dphi8B;
9
	full_text,
*
(%58 = phi i64 [ %53, %56 ], [ %66, %57 ]
%i648B

	full_text
	
i64 %53
%i648B

	full_text
	
i64 %66
Fphi8B=
;
	full_text.
,
*%59 = phi float [ %54, %56 ], [ %65, %57 ]
)float8B

	full_text

	float %54
)float8B

	full_text

	float %65
Dphi8B;
9
	full_text,
*
(%60 = phi i64 [ %17, %56 ], [ %67, %57 ]
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %67
6mul8B-
+
	full_text

%61 = mul nsw i64 %58, %13
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %13
6add8B-
+
	full_text

%62 = add nsw i64 %61, %14
%i648B

	full_text
	
i64 %61
%i648B

	full_text
	
i64 %14
\getelementptr8BI
G
	full_text:
8
6%63 = getelementptr inbounds float, float* %0, i64 %62
%i648B

	full_text
	
i64 %62
Lload8BB
@
	full_text3
1
/%64 = load float, float* %63, align 4, !tbaa !9
+float*8B

	full_text


float* %63
ecall8B[
Y
	full_textL
J
H%65 = tail call float @llvm.fmuladd.f32(float %64, float %64, float %59)
)float8B

	full_text

	float %64
)float8B

	full_text

	float %64
)float8B

	full_text

	float %59
8add8B/
-
	full_text 

%66 = add nuw nsw i64 %58, 1
%i648B

	full_text
	
i64 %58
1add8B(
&
	full_text

%67 = add i64 %60, -1
%i648B

	full_text
	
i64 %60
5icmp8B+
)
	full_text

%68 = icmp eq i64 %67, 0
%i648B

	full_text
	
i64 %67
Jbr8BB
@
	full_text3
1
/br i1 %68, label %69, label %57, !llvm.loop !13
#i18B

	full_text


i1 %68
]phi8BT
R
	full_textE
C
A%70 = phi float [ 0.000000e+00, %10 ], [ %52, %51 ], [ %65, %57 ]
)float8B

	full_text

	float %52
)float8B

	full_text

	float %65
Jcall8B@
>
	full_text1
/
-%71 = tail call float @_Z4sqrtf(float %70) #3
)float8B

	full_text

	float %70
4mul8B+
)
	full_text

%72 = mul nsw i32 %5, %3
5add8B,
*
	full_text

%73 = add nsw i32 %72, %3
%i328B

	full_text
	
i32 %72
6sext8B,
*
	full_text

%74 = sext i32 %73 to i64
%i328B

	full_text
	
i32 %73
\getelementptr8BI
G
	full_text:
8
6%75 = getelementptr inbounds float, float* %1, i64 %74
%i648B

	full_text
	
i64 %74
Lstore8BA
?
	full_text2
0
.store float %71, float* %75, align 4, !tbaa !9
)float8B

	full_text

	float %71
+float*8B

	full_text


float* %75
'br8B

	full_text

br label %76
$ret8	B

	full_text


ret void
$i328
B

	full_text


i32 %3
$i328
B

	full_text


i32 %5
*float*8
B

	full_text

	float* %0
*float*8
B

	full_text

	float* %1
$i328
B

	full_text


i32 %4
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
#i648
B

	full_text	

i64 2
2float8
B%
#
	full_text

float 0.000000e+00
#i648
B

	full_text	

i64 3
+float8
B

	full_text

float undef
#i648
B

	full_text	

i64 4
$i648
B

	full_text


i64 -1
#i648
B

	full_text	

i64 1
$i648
B

	full_text


i64 -4
#i328
B

	full_text	

i32 0
#i648
B

	full_text	

i64 0       	
 	                     !" !# !! $% $& $$ '( '' )* )) +, +- +. ++ /0 // 12 13 11 45 46 44 78 77 9: 99 ;< ;= ;> ;; ?@ ?? AB AC AA DE DF DD GH GG IJ II KL KM KN KK OP OO QR QS QQ TU TV TT WX WW YZ YY [\ [] [^ [[ _` __ ab aa cd cc ef eh gg ij ii kl kk mn mm op os rt rr uv uw uu xy xz xx {| {} {{ ~ ~	Ä ~~ Å
Ç ÅÅ ÉÑ ÉÉ ÖÜ Ö
á Ö
à ÖÖ âä ââ ãå ãã çé çç èê è
í ë
ì ëë îï îî ññ óò óó ôö ôô õ
ú õõ ùû ù
ü ùù †¢ 
¢ ñ
¢ ó£ £ ñ§ '§ 7§ G§ W§ Å• õ¶ ¶     
      _ [  a   " #! % &$ (' *) ,) - . 0/ 2 31 5 64 87 :9 <9 =+ > @? B CA E FD HG JI LI M; N PO R SQ U VT XW ZY \Y ]K ^ ` ba dc f[ h_ j[ l nm pi sâ tk vÖ w yã zr | }{  Ä~ ÇÅ ÑÉ ÜÉ áu àr äx åã éç êg íÖ ìë ïñ òó öô úî ûõ ü  °	 	 ë g † °o ëo q q re ge è ëè r ° ßß ®® ©©; ®® ;+ ®® +K ®® K[ ®® [Ö ®® Öî ©© î ßß 	™ ?´ ´ k´ ë	¨ 	¨ 	¨ O≠ g	Æ _	Ø 
Ø ã	∞ /
∞ â	± a≤ 	≤ 	≤ ≥ 	≥ c≥ i	≥ m
≥ ç"
gramschmidt_kernel1"
_Z13get_global_idj"
llvm.fmuladd.f32"

_Z4sqrtf*≠
4polybench-gpu-1.0-gramschmidt-gramschmidt_kernel1.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

wgsize
Ä

devmap_label


transfer_bytes
ÄÄÄ0

wgsize_log1p
kìA
 
transfer_bytes_log1p
kìA