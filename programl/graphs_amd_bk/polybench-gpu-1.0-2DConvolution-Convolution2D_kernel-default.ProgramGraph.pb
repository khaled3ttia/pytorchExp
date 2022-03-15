

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 0) #3
4truncB+
)
	full_text

%6 = trunc i64 %5 to i32
"i64B

	full_text


i64 %5
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 1) #3
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
1addB*
(
	full_text

%9 = add nsw i32 %2, -1
4icmpB,
*
	full_text

%10 = icmp sgt i32 %9, %8
"i32B

	full_text


i32 %9
"i32B

	full_text


i32 %8
8brB2
0
	full_text#
!
br i1 %10, label %11, label %72
!i1B

	full_text


i1 %10
4add8B+
)
	full_text

%12 = add nsw i32 %3, -1
7icmp8B-
+
	full_text

%13 = icmp sgt i32 %12, %6
%i328B

	full_text
	
i32 %12
$i328B

	full_text


i32 %6
5icmp8B+
)
	full_text

%14 = icmp sgt i32 %8, 0
$i328B

	full_text


i32 %8
1and8B(
&
	full_text

%15 = and i1 %13, %14
#i18B

	full_text


i1 %13
#i18B

	full_text


i1 %14
5icmp8B+
)
	full_text

%16 = icmp sgt i32 %6, 0
$i328B

	full_text


i32 %6
1and8B(
&
	full_text

%17 = and i1 %16, %15
#i18B

	full_text


i1 %16
#i18B

	full_text


i1 %15
:br8B2
0
	full_text#
!
br i1 %17, label %18, label %72
#i18B

	full_text


i1 %17
4add8B+
)
	full_text

%19 = add nsw i32 %8, -1
$i328B

	full_text


i32 %8
5mul8B,
*
	full_text

%20 = mul nsw i32 %19, %3
%i328B

	full_text
	
i32 %19
4add8B+
)
	full_text

%21 = add nsw i32 %6, -1
$i328B

	full_text


i32 %6
6add8B-
+
	full_text

%22 = add nsw i32 %20, %21
%i328B

	full_text
	
i32 %20
%i328B

	full_text
	
i32 %21
6sext8B,
*
	full_text

%23 = sext i32 %22 to i64
%i328B

	full_text
	
i32 %22
\getelementptr8BI
G
	full_text:
8
6%24 = getelementptr inbounds float, float* %0, i64 %23
%i648B

	full_text
	
i64 %23
Lload8BB
@
	full_text3
1
/%25 = load float, float* %24, align 4, !tbaa !9
+float*8B

	full_text


float* %24
5add8B,
*
	full_text

%26 = add nsw i32 %20, %6
%i328B

	full_text
	
i32 %20
$i328B

	full_text


i32 %6
6sext8B,
*
	full_text

%27 = sext i32 %26 to i64
%i328B

	full_text
	
i32 %26
\getelementptr8BI
G
	full_text:
8
6%28 = getelementptr inbounds float, float* %0, i64 %27
%i648B

	full_text
	
i64 %27
Lload8BB
@
	full_text3
1
/%29 = load float, float* %28, align 4, !tbaa !9
+float*8B

	full_text


float* %28
?fmul8B5
3
	full_text&
$
"%30 = fmul float %29, 5.000000e-01
)float8B

	full_text

	float %29
tcall8Bj
h
	full_text[
Y
W%31 = tail call float @llvm.fmuladd.f32(float %25, float 0x3FC99999A0000000, float %30)
)float8B

	full_text

	float %25
)float8B

	full_text

	float %30
3add8B*
(
	full_text

%32 = add nsw i32 %6, 1
$i328B

	full_text


i32 %6
6add8B-
+
	full_text

%33 = add nsw i32 %20, %32
%i328B

	full_text
	
i32 %20
%i328B

	full_text
	
i32 %32
6sext8B,
*
	full_text

%34 = sext i32 %33 to i64
%i328B

	full_text
	
i32 %33
\getelementptr8BI
G
	full_text:
8
6%35 = getelementptr inbounds float, float* %0, i64 %34
%i648B

	full_text
	
i64 %34
Lload8BB
@
	full_text3
1
/%36 = load float, float* %35, align 4, !tbaa !9
+float*8B

	full_text


float* %35
tcall8Bj
h
	full_text[
Y
W%37 = tail call float @llvm.fmuladd.f32(float %36, float 0xBFE99999A0000000, float %31)
)float8B

	full_text

	float %36
)float8B

	full_text

	float %31
4mul8B+
)
	full_text

%38 = mul nsw i32 %8, %3
$i328B

	full_text


i32 %8
6add8B-
+
	full_text

%39 = add nsw i32 %38, %21
%i328B

	full_text
	
i32 %38
%i328B

	full_text
	
i32 %21
6sext8B,
*
	full_text

%40 = sext i32 %39 to i64
%i328B

	full_text
	
i32 %39
\getelementptr8BI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %0, i64 %40
%i648B

	full_text
	
i64 %40
Lload8BB
@
	full_text3
1
/%42 = load float, float* %41, align 4, !tbaa !9
+float*8B

	full_text


float* %41
tcall8Bj
h
	full_text[
Y
W%43 = tail call float @llvm.fmuladd.f32(float %42, float 0xBFD3333340000000, float %37)
)float8B

	full_text

	float %42
)float8B

	full_text

	float %37
5add8B,
*
	full_text

%44 = add nsw i32 %38, %6
%i328B

	full_text
	
i32 %38
$i328B

	full_text


i32 %6
6sext8B,
*
	full_text

%45 = sext i32 %44 to i64
%i328B

	full_text
	
i32 %44
\getelementptr8BI
G
	full_text:
8
6%46 = getelementptr inbounds float, float* %0, i64 %45
%i648B

	full_text
	
i64 %45
Lload8BB
@
	full_text3
1
/%47 = load float, float* %46, align 4, !tbaa !9
+float*8B

	full_text


float* %46
tcall8Bj
h
	full_text[
Y
W%48 = tail call float @llvm.fmuladd.f32(float %47, float 0x3FE3333340000000, float %43)
)float8B

	full_text

	float %47
)float8B

	full_text

	float %43
6add8B-
+
	full_text

%49 = add nsw i32 %38, %32
%i328B

	full_text
	
i32 %38
%i328B

	full_text
	
i32 %32
6sext8B,
*
	full_text

%50 = sext i32 %49 to i64
%i328B

	full_text
	
i32 %49
\getelementptr8BI
G
	full_text:
8
6%51 = getelementptr inbounds float, float* %0, i64 %50
%i648B

	full_text
	
i64 %50
Lload8BB
@
	full_text3
1
/%52 = load float, float* %51, align 4, !tbaa !9
+float*8B

	full_text


float* %51
tcall8Bj
h
	full_text[
Y
W%53 = tail call float @llvm.fmuladd.f32(float %52, float 0xBFECCCCCC0000000, float %48)
)float8B

	full_text

	float %52
)float8B

	full_text

	float %48
3add8B*
(
	full_text

%54 = add nsw i32 %8, 1
$i328B

	full_text


i32 %8
5mul8B,
*
	full_text

%55 = mul nsw i32 %54, %3
%i328B

	full_text
	
i32 %54
6add8B-
+
	full_text

%56 = add nsw i32 %55, %21
%i328B

	full_text
	
i32 %55
%i328B

	full_text
	
i32 %21
6sext8B,
*
	full_text

%57 = sext i32 %56 to i64
%i328B

	full_text
	
i32 %56
\getelementptr8BI
G
	full_text:
8
6%58 = getelementptr inbounds float, float* %0, i64 %57
%i648B

	full_text
	
i64 %57
Lload8BB
@
	full_text3
1
/%59 = load float, float* %58, align 4, !tbaa !9
+float*8B

	full_text


float* %58
tcall8Bj
h
	full_text[
Y
W%60 = tail call float @llvm.fmuladd.f32(float %59, float 0x3FD99999A0000000, float %53)
)float8B

	full_text

	float %59
)float8B

	full_text

	float %53
5add8B,
*
	full_text

%61 = add nsw i32 %55, %6
%i328B

	full_text
	
i32 %55
$i328B

	full_text


i32 %6
6sext8B,
*
	full_text

%62 = sext i32 %61 to i64
%i328B

	full_text
	
i32 %61
\getelementptr8BI
G
	full_text:
8
6%63 = getelementptr inbounds float, float* %0, i64 %62
%i648B

	full_text
	
i64 %62
Lload8BB
@
	full_text3
1
/%64 = load float, float* %63, align 4, !tbaa !9
+float*8B

	full_text


float* %63
tcall8Bj
h
	full_text[
Y
W%65 = tail call float @llvm.fmuladd.f32(float %64, float 0x3FE6666660000000, float %60)
)float8B

	full_text

	float %64
)float8B

	full_text

	float %60
6add8B-
+
	full_text

%66 = add nsw i32 %55, %32
%i328B

	full_text
	
i32 %55
%i328B

	full_text
	
i32 %32
6sext8B,
*
	full_text

%67 = sext i32 %66 to i64
%i328B

	full_text
	
i32 %66
\getelementptr8BI
G
	full_text:
8
6%68 = getelementptr inbounds float, float* %0, i64 %67
%i648B

	full_text
	
i64 %67
Lload8BB
@
	full_text3
1
/%69 = load float, float* %68, align 4, !tbaa !9
+float*8B

	full_text


float* %68
tcall8Bj
h
	full_text[
Y
W%70 = tail call float @llvm.fmuladd.f32(float %69, float 0x3FB99999A0000000, float %65)
)float8B

	full_text

	float %69
)float8B

	full_text

	float %65
\getelementptr8BI
G
	full_text:
8
6%71 = getelementptr inbounds float, float* %1, i64 %45
%i648B

	full_text
	
i64 %45
Lstore8BA
?
	full_text2
0
.store float %70, float* %71, align 4, !tbaa !9
)float8B

	full_text

	float %70
+float*8B

	full_text


float* %71
'br8B

	full_text

br label %72
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %2
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %0
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
2float8B%
#
	full_text

float 5.000000e-01
8float8B+
)
	full_text

float 0x3FE6666660000000
8float8B+
)
	full_text

float 0xBFECCCCCC0000000
8float8B+
)
	full_text

float 0x3FB99999A0000000
8float8B+
)
	full_text

float 0x3FD99999A0000000
8float8B+
)
	full_text

float 0xBFD3333340000000
$i328B

	full_text


i32 -1
8float8B+
)
	full_text

float 0x3FC99999A0000000
#i328B

	full_text	

i32 1
8float8B+
)
	full_text

float 0x3FE3333340000000
#i328B

	full_text	

i32 0
8float8B+
)
	full_text

float 0xBFE99999A0000000       	 
                       !" !! #$ #% ## &' && () (( *+ ** ,- ,. ,, /0 // 12 11 34 33 56 55 78 79 77 :; :: <= <> << ?@ ?? AB AA CD CC EF EG EE HI HH JK JL JJ MN MM OP OO QR QQ ST SU SS VW VX VV YZ YY [\ [[ ]^ ]] _` _a __ bc bd bb ef ee gh gg ij ii kl km kk no nn pq pp rs rt rr uv uu wx ww yz yy {| {} {{ ~ ~	€ ~~ ‚  ƒ
„ ƒƒ …† …… ‡ˆ ‡
‰ ‡‡ Š‹ Š
Œ ŠŠ Ž  
  ‘’ ‘‘ “” “
• ““ –
— –– ˜™ ˜
š ˜˜ › 	 	 H	 pž Ÿ –  (  1  A  O  [  g  w  ƒ     	 
              " $! %# '& )( + - ., 0/ 21 43 6* 85 9 ; =: >< @? BA DC F7 G IH K! LJ NM PO RQ TE UH W XV ZY \[ ^] `S aH c: db fe hg ji l_ m on qp s! tr vu xw zy |k }p  €~ ‚ „ƒ †… ˆ{ ‰p ‹: ŒŠ Ž  ’‘ ”‡ •Y —“ ™– š  œ  œ› œ œ ¡¡ ¢¢ ¡¡  ¡¡ { ¢¢ {“ ¢¢ “k ¢¢ kS ¢¢ SE ¢¢ E‡ ¢¢ ‡7 ¢¢ 7_ ¢¢ _	£ 5
¤ ‡	¥ k
¦ “	§ {	¨ S	© 	© 	© 	© !	ª 7« 	« :	« n	¬ _­ 	­ 	­ 	® E"
Convolution2D_kernel"
_Z13get_global_idj"
llvm.fmuladd.f32*°
7polybench-gpu-1.0-2DConvolution-Convolution2D_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02

wgsize_log1p
D¸•A

devmap_label

 
transfer_bytes_log1p
D¸•A

transfer_bytes
€€€@

wgsize
€