

[external]
KcallBC
A
	full_text4
2
0%4 = tail call i64 @_Z13get_global_idj(i32 0) #3
4truncB+
)
	full_text

%5 = trunc i64 %4 to i32
"i64B

	full_text


i64 %4
3icmpB+
)
	full_text

%6 = icmp slt i32 %5, %2
"i32B

	full_text


i32 %5
6brB0
.
	full_text!

br i1 %6, label %7, label %12
 i1B

	full_text	

i1 %6
5trunc8B*
(
	full_text

%8 = trunc i16 %1 to i8
/shl8B&
$
	full_text

%9 = shl i64 %4, 32
$i648B

	full_text


i64 %4
8ashr8B.
,
	full_text

%10 = ashr exact i64 %9, 32
$i648B

	full_text


i64 %9
Vgetelementptr8BC
A
	full_text4
2
0%11 = getelementptr inbounds i8, i8* %0, i64 %10
%i648B

	full_text
	
i64 %10
Estore8B:
8
	full_text+
)
'store i8 %8, i8* %11, align 1, !tbaa !8
"i88B

	full_text	

i8 %8
%i8*8B

	full_text
	
i8* %11
'br8B

	full_text

br label %12
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %2
$i168B

	full_text


i16 %1
$i8*8B

	full_text


i8* %0
-; undefined function B

	full_text

 
Ocall 8BC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 0) #3
8trunc 8B+
)
	full_text

%8 = trunc i64 %7 to i32
&i64 8B

	full_text


i64 %7
7icmp 8B+
)
	full_text

%9 = icmp slt i32 %8, %1
&i32 8B

	full_text


i32 %8
;br 8B1
/
	full_text"
 
br i1 %9, label %10, label %59
$i1 8B

	full_text	

i1 %9
2shl 8B'
%
	full_text

%11 = shl i64 %7, 32
&i64 8B

	full_text


i64 %7
;ashr 8B/
-
	full_text 

%12 = ashr exact i64 %11, 32
'i64 8B

	full_text
	
i64 %11
^getelementptr 8BI
G
	full_text:
8
6%13 = getelementptr inbounds float, float* %4, i64 %12
'i64 8B

	full_text
	
i64 %12
Nload 8BB
@
	full_text3
1
/%14 = load float, float* %13, align 4, !tbaa !8
-float* 8B

	full_text


float* %13
5sub 8B*
(
	full_text

%15 = sub nsw i32 4, %0
>sitofp 8B0
.
	full_text!

%16 = sitofp i32 %15 to float
'i32 8B

	full_text
	
i32 %15
Efdiv 8B9
7
	full_text*
(
&%17 = fdiv float %14, %16, !fpmath !12
+float 8B

	full_text

	float %14
+float 8B

	full_text

	float %16
^getelementptr 8BI
G
	full_text:
8
6%18 = getelementptr inbounds float, float* %2, i64 %12
'i64 8B

	full_text
	
i64 %12
Nload 8BB
@
	full_text3
1
/%19 = load float, float* %18, align 4, !tbaa !8
-float* 8B

	full_text


float* %18
^getelementptr 8BI
G
	full_text:
8
6%20 = getelementptr inbounds float, float* %5, i64 %12
'i64 8B

	full_text
	
i64 %12
Nload 8BB
@
	full_text3
1
/%21 = load float, float* %20, align 4, !tbaa !8
-float* 8B

	full_text


float* %20
gcall 8B[
Y
	full_textL
J
H%22 = tail call float @llvm.fmuladd.f32(float %17, float %21, float %19)
+float 8B

	full_text

	float %17
+float 8B

	full_text

	float %21
+float 8B

	full_text

	float %19
^getelementptr 8BI
G
	full_text:
8
6%23 = getelementptr inbounds float, float* %3, i64 %12
'i64 8B

	full_text
	
i64 %12
Nstore 8BA
?
	full_text2
0
.store float %22, float* %23, align 4, !tbaa !8
+float 8B

	full_text

	float %22
-float* 8B

	full_text


float* %23
1shl 8B&
$
	full_text

%24 = shl i32 %1, 2
7add 8B,
*
	full_text

%25 = add nsw i32 %24, %8
'i32 8B

	full_text
	
i32 %24
&i32 8B

	full_text


i32 %8
8sext 8B,
*
	full_text

%26 = sext i32 %25 to i64
'i32 8B

	full_text
	
i32 %25
^getelementptr 8BI
G
	full_text:
8
6%27 = getelementptr inbounds float, float* %2, i64 %26
'i64 8B

	full_text
	
i64 %26
Nload 8BB
@
	full_text3
1
/%28 = load float, float* %27, align 4, !tbaa !8
-float* 8B

	full_text


float* %27
^getelementptr 8BI
G
	full_text:
8
6%29 = getelementptr inbounds float, float* %5, i64 %26
'i64 8B

	full_text
	
i64 %26
Nload 8BB
@
	full_text3
1
/%30 = load float, float* %29, align 4, !tbaa !8
-float* 8B

	full_text


float* %29
gcall 8B[
Y
	full_textL
J
H%31 = tail call float @llvm.fmuladd.f32(float %17, float %30, float %28)
+float 8B

	full_text

	float %17
+float 8B

	full_text

	float %30
+float 8B

	full_text

	float %28
^getelementptr 8BI
G
	full_text:
8
6%32 = getelementptr inbounds float, float* %3, i64 %26
'i64 8B

	full_text
	
i64 %26
Nstore 8BA
?
	full_text2
0
.store float %31, float* %32, align 4, !tbaa !8
+float 8B

	full_text

	float %31
-float* 8B

	full_text


float* %32
6add 8B+
)
	full_text

%33 = add nsw i32 %8, %1
&i32 8B

	full_text


i32 %8
8sext 8B,
*
	full_text

%34 = sext i32 %33 to i64
'i32 8B

	full_text
	
i32 %33
^getelementptr 8BI
G
	full_text:
8
6%35 = getelementptr inbounds float, float* %2, i64 %34
'i64 8B

	full_text
	
i64 %34
Nload 8BB
@
	full_text3
1
/%36 = load float, float* %35, align 4, !tbaa !8
-float* 8B

	full_text


float* %35
^getelementptr 8BI
G
	full_text:
8
6%37 = getelementptr inbounds float, float* %5, i64 %34
'i64 8B

	full_text
	
i64 %34
Nload 8BB
@
	full_text3
1
/%38 = load float, float* %37, align 4, !tbaa !8
-float* 8B

	full_text


float* %37
gcall 8B[
Y
	full_textL
J
H%39 = tail call float @llvm.fmuladd.f32(float %17, float %38, float %36)
+float 8B

	full_text

	float %17
+float 8B

	full_text

	float %38
+float 8B

	full_text

	float %36
^getelementptr 8BI
G
	full_text:
8
6%40 = getelementptr inbounds float, float* %3, i64 %34
'i64 8B

	full_text
	
i64 %34
Nstore 8BA
?
	full_text2
0
.store float %39, float* %40, align 4, !tbaa !8
+float 8B

	full_text

	float %39
-float* 8B

	full_text


float* %40
1shl 8B&
$
	full_text

%41 = shl i32 %1, 1
7add 8B,
*
	full_text

%42 = add nsw i32 %41, %8
'i32 8B

	full_text
	
i32 %41
&i32 8B

	full_text


i32 %8
8sext 8B,
*
	full_text

%43 = sext i32 %42 to i64
'i32 8B

	full_text
	
i32 %42
^getelementptr 8BI
G
	full_text:
8
6%44 = getelementptr inbounds float, float* %2, i64 %43
'i64 8B

	full_text
	
i64 %43
Nload 8BB
@
	full_text3
1
/%45 = load float, float* %44, align 4, !tbaa !8
-float* 8B

	full_text


float* %44
^getelementptr 8BI
G
	full_text:
8
6%46 = getelementptr inbounds float, float* %5, i64 %43
'i64 8B

	full_text
	
i64 %43
Nload 8BB
@
	full_text3
1
/%47 = load float, float* %46, align 4, !tbaa !8
-float* 8B

	full_text


float* %46
gcall 8B[
Y
	full_textL
J
H%48 = tail call float @llvm.fmuladd.f32(float %17, float %47, float %45)
+float 8B

	full_text

	float %17
+float 8B

	full_text

	float %47
+float 8B

	full_text

	float %45
^getelementptr 8BI
G
	full_text:
8
6%49 = getelementptr inbounds float, float* %3, i64 %43
'i64 8B

	full_text
	
i64 %43
Nstore 8BA
?
	full_text2
0
.store float %48, float* %49, align 4, !tbaa !8
+float 8B

	full_text

	float %48
-float* 8B

	full_text


float* %49
5mul 8B*
(
	full_text

%50 = mul nsw i32 %1, 3
7add 8B,
*
	full_text

%51 = add nsw i32 %50, %8
'i32 8B

	full_text
	
i32 %50
&i32 8B

	full_text


i32 %8
8sext 8B,
*
	full_text

%52 = sext i32 %51 to i64
'i32 8B

	full_text
	
i32 %51
^getelementptr 8BI
G
	full_text:
8
6%53 = getelementptr inbounds float, float* %2, i64 %52
'i64 8B

	full_text
	
i64 %52
Nload 8BB
@
	full_text3
1
/%54 = load float, float* %53, align 4, !tbaa !8
-float* 8B

	full_text


float* %53
^getelementptr 8BI
G
	full_text:
8
6%55 = getelementptr inbounds float, float* %5, i64 %52
'i64 8B

	full_text
	
i64 %52
Nload 8BB
@
	full_text3
1
/%56 = load float, float* %55, align 4, !tbaa !8
-float* 8B

	full_text


float* %55
gcall 8B[
Y
	full_textL
J
H%57 = tail call float @llvm.fmuladd.f32(float %17, float %56, float %54)
+float 8B

	full_text

	float %17
+float 8B

	full_text

	float %56
+float 8B

	full_text

	float %54
^getelementptr 8BI
G
	full_text:
8
6%58 = getelementptr inbounds float, float* %3, i64 %52
'i64 8B

	full_text
	
i64 %52
Nstore 8BA
?
	full_text2
0
.store float %57, float* %58, align 4, !tbaa !8
+float 8B

	full_text

	float %57
-float* 8B

	full_text


float* %58
)br 8B

	full_text

br label %59
&ret 8B

	full_text


ret void
&i32 8B

	full_text


i32 %1
,float* 8B

	full_text

	float* %2
,float* 8B

	full_text

	float* %3
,float* 8B

	full_text

	float* %4
&i32 8B

	full_text


i32 %0
,float* 8B

	full_text

	float* %5
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 4
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 3
#i328B

	full_text	

i32 2       	
 		               
	                !" !! #$ ## %& %% '' () (( *+ *, ** -. -- /0 // 12 11 34 33 56 57 58 55 9: 99 ;< ;= ;; >> ?@ ?A ?? BC BB DE DD FG FF HI HH JK JJ LM LN LO LL PQ PP RS RT RR UV UU WX WW YZ YY [\ [[ ]^ ]] _` __ ab ac ad aa ef ee gh gi gg jj kl km kk no nn pq pp rs rr tu tt vw vv xy xz x{ xx |} || ~ ~	? ~~ ?? ?? ?
? ?? ?? ?? ?
? ?? ?? ?? ?
? ?? ?? ?? ?? ?
? ?
? ?? ?
? ?? ?? ?
? ?? ?	? ? >	? U? j? ?? -? D? Y? p? ?? 9? P? e? |? ?? #	? '? 1? H? ]? t? ?      "! $# &' )% +( ,! .- 0! 21 4* 63 7/ 8! :5 <9 => @ A? CB ED GB IH K* MJ NF OB QL SP T VU XW ZY \W ^] `* b_ c[ dW fa he ij l mk on qp sn ut w* yv zr {n }x | ?? ? ?? ?? ?? ?? ?? ?* ?? ?? ?? ?? ?? ?  ?? ?  ?  ??5 ?? 5x ?? x    L ?? La ?? a? ?? ?? ? ? '	? j	? 		? 	? 	? !
? ?	? >"
memset_kernel"
_Z13get_global_idj"
	time_step"
llvm.fmuladd.f32*?
rodinia-3.1-cfd-time_step.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

wgsize_log1p
I??A
 
transfer_bytes_log1p
I??A

devmap_label
 

transfer_bytes
???

wgsize
?