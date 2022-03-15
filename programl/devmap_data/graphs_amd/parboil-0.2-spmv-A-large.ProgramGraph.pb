

[external]
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 0) #3
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
5icmpB-
+
	full_text

%11 = icmp slt i32 %10, %5
#i32B

	full_text
	
i32 %10
8brB2
0
	full_text#
!
br i1 %11, label %12, label %82
!i1B

	full_text


i1 %11
3sdiv8B)
'
	full_text

%13 = sdiv i32 %10, 32
%i328B

	full_text
	
i32 %10
6sext8B,
*
	full_text

%14 = sext i32 %13 to i64
%i328B

	full_text
	
i32 %13
Xgetelementptr8BE
C
	full_text6
4
2%15 = getelementptr inbounds i32, i32* %7, i64 %14
%i648B

	full_text
	
i64 %14
Hload8B>
<
	full_text/
-
+%16 = load i32, i32* %15, align 4, !tbaa !8
'i32*8B

	full_text


i32* %15
6icmp8B,
*
	full_text

%17 = icmp sgt i32 %16, 0
%i328B

	full_text
	
i32 %16
:br8B2
0
	full_text#
!
br i1 %17, label %18, label %42
#i18B

	full_text


i1 %17
6zext8B,
*
	full_text

%19 = zext i32 %16 to i64
%i328B

	full_text
	
i32 %16
0and8B'
%
	full_text

%20 = and i64 %19, 1
%i648B

	full_text
	
i64 %19
5icmp8B+
)
	full_text

%21 = icmp eq i32 %16, 1
%i328B

	full_text
	
i32 %16
:br8B2
0
	full_text#
!
br i1 %21, label %24, label %22
#i18B

	full_text


i1 %21
6sub8B-
+
	full_text

%23 = sub nsw i64 %19, %20
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %20
'br8B

	full_text

br label %50
Hphi8B?
=
	full_text0
.
,%25 = phi float [ undef, %18 ], [ %78, %50 ]
)float8B

	full_text

	float %78
Bphi8B9
7
	full_text*
(
&%26 = phi i64 [ 0, %18 ], [ %79, %50 ]
%i648B

	full_text
	
i64 %79
Ophi8BF
D
	full_text7
5
3%27 = phi float [ 0.000000e+00, %18 ], [ %78, %50 ]
)float8B

	full_text

	float %78
5icmp8B+
)
	full_text

%28 = icmp eq i64 %20, 0
%i648B

	full_text
	
i64 %20
:br8B2
0
	full_text#
!
br i1 %28, label %42, label %29
#i18B

	full_text


i1 %28
Xgetelementptr8BE
C
	full_text6
4
2%30 = getelementptr inbounds i32, i32* %6, i64 %26
%i648B

	full_text
	
i64 %26
Hload8B>
<
	full_text/
-
+%31 = load i32, i32* %30, align 4, !tbaa !8
'i32*8B

	full_text


i32* %30
6add8B-
+
	full_text

%32 = add nsw i32 %31, %10
%i328B

	full_text
	
i32 %31
%i328B

	full_text
	
i32 %10
6sext8B,
*
	full_text

%33 = sext i32 %32 to i64
%i328B

	full_text
	
i32 %32
\getelementptr8BI
G
	full_text:
8
6%34 = getelementptr inbounds float, float* %1, i64 %33
%i648B

	full_text
	
i64 %33
Mload8BC
A
	full_text4
2
0%35 = load float, float* %34, align 4, !tbaa !12
+float*8B

	full_text


float* %34
Xgetelementptr8BE
C
	full_text6
4
2%36 = getelementptr inbounds i32, i32* %2, i64 %33
%i648B

	full_text
	
i64 %33
Hload8B>
<
	full_text/
-
+%37 = load i32, i32* %36, align 4, !tbaa !8
'i32*8B

	full_text


i32* %36
6sext8B,
*
	full_text

%38 = sext i32 %37 to i64
%i328B

	full_text
	
i32 %37
\getelementptr8BI
G
	full_text:
8
6%39 = getelementptr inbounds float, float* %4, i64 %38
%i648B

	full_text
	
i64 %38
Mload8BC
A
	full_text4
2
0%40 = load float, float* %39, align 4, !tbaa !12
+float*8B

	full_text


float* %39
ecall8B[
Y
	full_textL
J
H%41 = tail call float @llvm.fmuladd.f32(float %35, float %40, float %27)
)float8B

	full_text

	float %35
)float8B

	full_text

	float %40
)float8B

	full_text

	float %27
'br8B

	full_text

br label %42
]phi8BT
R
	full_textE
C
A%43 = phi float [ 0.000000e+00, %12 ], [ %25, %24 ], [ %41, %29 ]
)float8B

	full_text

	float %25
)float8B

	full_text

	float %41
0shl8B'
%
	full_text

%44 = shl i64 %9, 32
$i648B

	full_text


i64 %9
9ashr8B/
-
	full_text 

%45 = ashr exact i64 %44, 32
%i648B

	full_text
	
i64 %44
Xgetelementptr8BE
C
	full_text6
4
2%46 = getelementptr inbounds i32, i32* %3, i64 %45
%i648B

	full_text
	
i64 %45
Hload8B>
<
	full_text/
-
+%47 = load i32, i32* %46, align 4, !tbaa !8
'i32*8B

	full_text


i32* %46
6sext8B,
*
	full_text

%48 = sext i32 %47 to i64
%i328B

	full_text
	
i32 %47
\getelementptr8BI
G
	full_text:
8
6%49 = getelementptr inbounds float, float* %0, i64 %48
%i648B

	full_text
	
i64 %48
Mstore8BB
@
	full_text3
1
/store float %43, float* %49, align 4, !tbaa !12
)float8B

	full_text

	float %43
+float*8B

	full_text


float* %49
'br8B

	full_text

br label %82
Bphi8B9
7
	full_text*
(
&%51 = phi i64 [ 0, %22 ], [ %79, %50 ]
%i648B

	full_text
	
i64 %79
Ophi8BF
D
	full_text7
5
3%52 = phi float [ 0.000000e+00, %22 ], [ %78, %50 ]
)float8B

	full_text

	float %78
Dphi8B;
9
	full_text,
*
(%53 = phi i64 [ %23, %22 ], [ %80, %50 ]
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %80
Xgetelementptr8BE
C
	full_text6
4
2%54 = getelementptr inbounds i32, i32* %6, i64 %51
%i648B

	full_text
	
i64 %51
Hload8B>
<
	full_text/
-
+%55 = load i32, i32* %54, align 4, !tbaa !8
'i32*8B

	full_text


i32* %54
6add8B-
+
	full_text

%56 = add nsw i32 %55, %10
%i328B

	full_text
	
i32 %55
%i328B

	full_text
	
i32 %10
6sext8B,
*
	full_text

%57 = sext i32 %56 to i64
%i328B

	full_text
	
i32 %56
Xgetelementptr8BE
C
	full_text6
4
2%58 = getelementptr inbounds i32, i32* %2, i64 %57
%i648B

	full_text
	
i64 %57
Hload8B>
<
	full_text/
-
+%59 = load i32, i32* %58, align 4, !tbaa !8
'i32*8B

	full_text


i32* %58
\getelementptr8BI
G
	full_text:
8
6%60 = getelementptr inbounds float, float* %1, i64 %57
%i648B

	full_text
	
i64 %57
Mload8BC
A
	full_text4
2
0%61 = load float, float* %60, align 4, !tbaa !12
+float*8B

	full_text


float* %60
6sext8B,
*
	full_text

%62 = sext i32 %59 to i64
%i328B

	full_text
	
i32 %59
\getelementptr8BI
G
	full_text:
8
6%63 = getelementptr inbounds float, float* %4, i64 %62
%i648B

	full_text
	
i64 %62
Mload8BC
A
	full_text4
2
0%64 = load float, float* %63, align 4, !tbaa !12
+float*8B

	full_text


float* %63
ecall8B[
Y
	full_textL
J
H%65 = tail call float @llvm.fmuladd.f32(float %61, float %64, float %52)
)float8B

	full_text

	float %61
)float8B

	full_text

	float %64
)float8B

	full_text

	float %52
.or8B&
$
	full_text

%66 = or i64 %51, 1
%i648B

	full_text
	
i64 %51
Xgetelementptr8BE
C
	full_text6
4
2%67 = getelementptr inbounds i32, i32* %6, i64 %66
%i648B

	full_text
	
i64 %66
Hload8B>
<
	full_text/
-
+%68 = load i32, i32* %67, align 4, !tbaa !8
'i32*8B

	full_text


i32* %67
6add8B-
+
	full_text

%69 = add nsw i32 %68, %10
%i328B

	full_text
	
i32 %68
%i328B

	full_text
	
i32 %10
6sext8B,
*
	full_text

%70 = sext i32 %69 to i64
%i328B

	full_text
	
i32 %69
Xgetelementptr8BE
C
	full_text6
4
2%71 = getelementptr inbounds i32, i32* %2, i64 %70
%i648B

	full_text
	
i64 %70
Hload8B>
<
	full_text/
-
+%72 = load i32, i32* %71, align 4, !tbaa !8
'i32*8B

	full_text


i32* %71
\getelementptr8BI
G
	full_text:
8
6%73 = getelementptr inbounds float, float* %1, i64 %70
%i648B

	full_text
	
i64 %70
Mload8BC
A
	full_text4
2
0%74 = load float, float* %73, align 4, !tbaa !12
+float*8B

	full_text


float* %73
6sext8B,
*
	full_text

%75 = sext i32 %72 to i64
%i328B

	full_text
	
i32 %72
\getelementptr8BI
G
	full_text:
8
6%76 = getelementptr inbounds float, float* %4, i64 %75
%i648B

	full_text
	
i64 %75
Mload8BC
A
	full_text4
2
0%77 = load float, float* %76, align 4, !tbaa !12
+float*8B

	full_text


float* %76
ecall8B[
Y
	full_textL
J
H%78 = tail call float @llvm.fmuladd.f32(float %74, float %77, float %65)
)float8B

	full_text

	float %74
)float8B

	full_text

	float %77
)float8B

	full_text

	float %65
4add8B+
)
	full_text

%79 = add nsw i64 %51, 2
%i648B

	full_text
	
i64 %51
1add8B(
&
	full_text

%80 = add i64 %53, -2
%i648B

	full_text
	
i64 %53
5icmp8B+
)
	full_text

%81 = icmp eq i64 %80, 0
%i648B

	full_text
	
i64 %80
:br8B2
0
	full_text#
!
br i1 %81, label %24, label %50
#i18B

	full_text


i1 %81
$ret8B

	full_text


ret void
*float*8	B

	full_text

	float* %1
*float*8	B

	full_text

	float* %4
&i32*8	B

	full_text
	
i32* %7
*float*8	B

	full_text

	float* %0
&i32*8	B

	full_text
	
i32* %6
$i328	B

	full_text


i32 %5
&i32*8	B

	full_text
	
i32* %3
&i32*8	B

	full_text
	
i32* %2
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
$i648	B

	full_text


i64 -2
2float8	B%
#
	full_text

float 0.000000e+00
#i328	B

	full_text	

i32 0
+float8	B

	full_text

float undef
#i328	B

	full_text	

i32 1
#i648	B

	full_text	

i64 1
#i648	B

	full_text	

i64 2
#i648	B

	full_text	

i64 0
$i328	B

	full_text


i32 32
$i648	B

	full_text


i64 32      	  
 

                  !    "# "" $% $$ &' && () (+ ** ,- ,, ./ .0 .. 12 11 34 33 56 55 78 77 9: 99 ;< ;; => == ?@ ?? AB AC AD AA EG FH FF IJ II KL KK MN MM OP OO QR QQ ST SS UV UW UU XZ YY [\ [[ ]^ ]_ ]] `a `` bc bb de df dd gh gg ij ii kl kk mn mm op oo qr qq st ss uv uu wx wy wz ww {| {{ }~ }} Ä  ÅÇ Å
É ÅÅ ÑÖ ÑÑ Ü
á ÜÜ àâ àà ä
ã ää åç åå éè éé ê
ë êê íì íí îï î
ñ î
ó îî òô òò öõ öö úù úú ûü û° 3° m° ä¢ =¢ s¢ ê£ § S• *• `• }	¶ ß M® 7® i® Ü    	 
          î !ò #î % '& )" +* -, / 0. 21 43 61 87 :9 <; >= @5 B? C$ D  GA H JI LK NM PO RQ TF VS Wò Zî \ ^ö _Y a` cb e fd hg ji lg nm pk rq ts vo xu y[ zY |{ ~} Ä Ç ÉÅ ÖÑ áÜ âÑ ãä çà èé ëê ìå ïí ñw óY ô] õö ùú ü  †  F   X †( F( * YE Fû  û Y ©© † ™™A ™™ Aw ™™ w ©© î ™™ î
´ ö¨ $¨ F¨ [≠ 	≠ Æ  	Ø 	∞ 	∞ {
± ò≤ "	≤ &≤ Y
≤ ú	≥ 	¥ I	¥ K"
spmv_jds_naive"
_Z13get_global_idj"
llvm.fmuladd.f32*é
parboil-0.2-spmv-A.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

devmap_label
 

wgsize_log1p
v¯âA
 
transfer_bytes_log1p
v¯âA

transfer_bytes
Ãäﬁ

wgsize
¿