
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
5icmpB-
+
	full_text

%11 = icmp slt i32 %10, %3
#i32B

	full_text
	
i32 %10
4icmpB,
*
	full_text

%12 = icmp slt i32 %8, %4
"i32B

	full_text


i32 %8
/andB(
&
	full_text

%13 = and i1 %12, %11
!i1B

	full_text


i1 %12
!i1B

	full_text


i1 %11
8brB2
0
	full_text#
!
br i1 %13, label %14, label %68
!i1B

	full_text


i1 %13
5mul8B,
*
	full_text

%15 = mul nsw i32 %10, %4
%i328B

	full_text
	
i32 %10
5add8B,
*
	full_text

%16 = add nsw i32 %15, %8
%i328B

	full_text
	
i32 %15
$i328B

	full_text


i32 %8
6sext8B,
*
	full_text

%17 = sext i32 %16 to i64
%i328B

	full_text
	
i32 %16
\getelementptr8BI
G
	full_text:
8
6%18 = getelementptr inbounds float, float* %2, i64 %17
%i648B

	full_text
	
i64 %17
Ustore8BJ
H
	full_text;
9
7store float 0.000000e+00, float* %18, align 4, !tbaa !9
+float*8B

	full_text


float* %18
5icmp8B+
)
	full_text

%19 = icmp sgt i32 %5, 0
:br8B2
0
	full_text#
!
br i1 %19, label %20, label %68
#i18B

	full_text


i1 %19
5mul8B,
*
	full_text

%21 = mul nsw i32 %10, %5
%i328B

	full_text
	
i32 %10
5sext8B+
)
	full_text

%22 = sext i32 %4 to i64
0shl8B'
%
	full_text

%23 = shl i64 %7, 32
$i648B

	full_text


i64 %7
9ashr8B/
-
	full_text 

%24 = ashr exact i64 %23, 32
%i648B

	full_text
	
i64 %23
6sext8B,
*
	full_text

%25 = sext i32 %21 to i64
%i328B

	full_text
	
i32 %21
5zext8B+
)
	full_text

%26 = zext i32 %5 to i64
0and8B'
%
	full_text

%27 = and i64 %26, 1
%i648B

	full_text
	
i64 %26
4icmp8B*
(
	full_text

%28 = icmp eq i32 %5, 1
:br8B2
0
	full_text#
!
br i1 %28, label %55, label %29
#i18B

	full_text


i1 %28
6sub8B-
+
	full_text

%30 = sub nsw i64 %26, %27
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %27
'br8B

	full_text

br label %31
Ophi8BF
D
	full_text7
5
3%32 = phi float [ 0.000000e+00, %29 ], [ %51, %31 ]
)float8B

	full_text

	float %51
Bphi8B9
7
	full_text*
(
&%33 = phi i64 [ 0, %29 ], [ %52, %31 ]
%i648B

	full_text
	
i64 %52
Dphi8B;
9
	full_text,
*
(%34 = phi i64 [ %30, %29 ], [ %53, %31 ]
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %53
6add8B-
+
	full_text

%35 = add nsw i64 %33, %25
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %25
\getelementptr8BI
G
	full_text:
8
6%36 = getelementptr inbounds float, float* %0, i64 %35
%i648B

	full_text
	
i64 %35
Lload8BB
@
	full_text3
1
/%37 = load float, float* %36, align 4, !tbaa !9
+float*8B

	full_text


float* %36
6mul8B-
+
	full_text

%38 = mul nsw i64 %33, %22
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %22
6add8B-
+
	full_text

%39 = add nsw i64 %38, %24
%i648B

	full_text
	
i64 %38
%i648B

	full_text
	
i64 %24
\getelementptr8BI
G
	full_text:
8
6%40 = getelementptr inbounds float, float* %1, i64 %39
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
ecall8B[
Y
	full_textL
J
H%42 = tail call float @llvm.fmuladd.f32(float %37, float %41, float %32)
)float8B

	full_text

	float %37
)float8B

	full_text

	float %41
)float8B

	full_text

	float %32
Lstore8BA
?
	full_text2
0
.store float %42, float* %18, align 4, !tbaa !9
)float8B

	full_text

	float %42
+float*8B

	full_text


float* %18
.or8B&
$
	full_text

%43 = or i64 %33, 1
%i648B

	full_text
	
i64 %33
6add8B-
+
	full_text

%44 = add nsw i64 %43, %25
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %25
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
6mul8B-
+
	full_text

%47 = mul nsw i64 %43, %22
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %22
6add8B-
+
	full_text

%48 = add nsw i64 %47, %24
%i648B

	full_text
	
i64 %47
%i648B

	full_text
	
i64 %24
\getelementptr8BI
G
	full_text:
8
6%49 = getelementptr inbounds float, float* %1, i64 %48
%i648B

	full_text
	
i64 %48
Lload8BB
@
	full_text3
1
/%50 = load float, float* %49, align 4, !tbaa !9
+float*8B

	full_text


float* %49
ecall8B[
Y
	full_textL
J
H%51 = tail call float @llvm.fmuladd.f32(float %46, float %50, float %42)
)float8B

	full_text

	float %46
)float8B

	full_text

	float %50
)float8B

	full_text

	float %42
Lstore8BA
?
	full_text2
0
.store float %51, float* %18, align 4, !tbaa !9
)float8B

	full_text

	float %51
+float*8B

	full_text


float* %18
4add8B+
)
	full_text

%52 = add nsw i64 %33, 2
%i648B

	full_text
	
i64 %33
1add8B(
&
	full_text

%53 = add i64 %34, -2
%i648B

	full_text
	
i64 %34
5icmp8B+
)
	full_text

%54 = icmp eq i64 %53, 0
%i648B

	full_text
	
i64 %53
:br8B2
0
	full_text#
!
br i1 %54, label %55, label %31
#i18B

	full_text


i1 %54
Ophi8BF
D
	full_text7
5
3%56 = phi float [ 0.000000e+00, %20 ], [ %51, %31 ]
)float8B

	full_text

	float %51
Bphi8B9
7
	full_text*
(
&%57 = phi i64 [ 0, %20 ], [ %52, %31 ]
%i648B

	full_text
	
i64 %52
5icmp8B+
)
	full_text

%58 = icmp eq i64 %27, 0
%i648B

	full_text
	
i64 %27
:br8B2
0
	full_text#
!
br i1 %58, label %68, label %59
#i18B

	full_text


i1 %58
6add8B-
+
	full_text

%60 = add nsw i64 %57, %25
%i648B

	full_text
	
i64 %57
%i648B

	full_text
	
i64 %25
\getelementptr8BI
G
	full_text:
8
6%61 = getelementptr inbounds float, float* %0, i64 %60
%i648B

	full_text
	
i64 %60
Lload8BB
@
	full_text3
1
/%62 = load float, float* %61, align 4, !tbaa !9
+float*8B

	full_text


float* %61
6mul8B-
+
	full_text

%63 = mul nsw i64 %57, %22
%i648B

	full_text
	
i64 %57
%i648B

	full_text
	
i64 %22
6add8B-
+
	full_text

%64 = add nsw i64 %63, %24
%i648B

	full_text
	
i64 %63
%i648B

	full_text
	
i64 %24
\getelementptr8BI
G
	full_text:
8
6%65 = getelementptr inbounds float, float* %1, i64 %64
%i648B

	full_text
	
i64 %64
Lload8BB
@
	full_text3
1
/%66 = load float, float* %65, align 4, !tbaa !9
+float*8B

	full_text


float* %65
ecall8B[
Y
	full_textL
J
H%67 = tail call float @llvm.fmuladd.f32(float %62, float %66, float %56)
)float8B

	full_text

	float %62
)float8B

	full_text

	float %66
)float8B

	full_text

	float %56
Lstore8BA
?
	full_text2
0
.store float %67, float* %18, align 4, !tbaa !9
)float8B

	full_text

	float %67
+float*8B

	full_text


float* %18
'br8B

	full_text

br label %68
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %0
*float*8B

	full_text

	float* %1
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %4
*float*8B

	full_text

	float* %2
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
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 2
2float8B%
#
	full_text

float 0.000000e+00
#i648B

	full_text	

i64 0
$i648B

	full_text


i64 -2
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 1        	
 		                       !" !! #$ ## %& %% '' () (( ** +, +. -/ -- 02 11 34 33 56 57 55 89 8: 88 ;< ;; => == ?@ ?A ?? BC BD BB EF EE GH GG IJ IK IL II MN MO MM PQ PP RS RT RR UV UU WX WW YZ Y[ YY \] \^ \\ _` __ ab aa cd ce cf cc gh gi gg jk jj lm ll no nn pq ps rr tu tt vw vv xy x{ z| zz }~ }} ?  ?? ?
? ?? ?? ?
? ?? ?
? ?? ?? ?? ?? ?
? ?
? ?? ?? ?
? ?? ?	? ? ;? U? }? E? _? ?? 	? ? '? *	? 		? ?  ?     
	            "! $ &' )* ,' .( /c 2j 4- 6l 73 9% :8 <; >3 @  A? C# DB FE H= JG K1 LI N O3 QP S% TR VU XP Z  [Y ]# ^\ `_ bW da eI fc h i3 k5 ml on qc sj u( wv yt {% |z ~} ?t ?  ?? ?# ?? ?? ? ?? ?r ?? ? ?  ?  ?+ r+ -x ?x z0 1? ?p rp 1 ? ?? ?? ?? I ?? Ic ?? c? ?? ? ?? 	? !	? #	? j? ? 1? r? 3	? n? t	? v	? l? 	? 	? (	? P? 	? *"
mm3_kernel2"
_Z13get_global_idj"
llvm.fmuladd.f32*?
$polybench-gpu-1.0-3mm-mm3_kernel2.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?
 
transfer_bytes_log1p
?|A

wgsize_log1p
?|A

devmap_label


wgsize
?

transfer_bytes
???