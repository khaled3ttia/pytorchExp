

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 0) #3
4truncB+
)
	full_text

%7 = trunc i64 %6 to i32
"i64B

	full_text


i64 %6
3icmpB+
)
	full_text

%8 = icmp slt i32 %7, %4
"i32B

	full_text


i32 %7
6brB0
.
	full_text!

br i1 %8, label %9, label %57
 i1B

	full_text	

i1 %8
0shl8B'
%
	full_text

%10 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%11 = ashr exact i64 %10, 32
%i648B

	full_text
	
i64 %10
\getelementptr8BI
G
	full_text:
8
6%12 = getelementptr inbounds float, float* %2, i64 %11
%i648B

	full_text
	
i64 %11
Ustore8BJ
H
	full_text;
9
7store float 0.000000e+00, float* %12, align 4, !tbaa !9
+float*8B

	full_text


float* %12
5icmp8B+
)
	full_text

%13 = icmp sgt i32 %3, 0
:br8B2
0
	full_text#
!
br i1 %13, label %14, label %57
#i18B

	full_text


i1 %13
5sext8B+
)
	full_text

%15 = sext i32 %4 to i64
0shl8B'
%
	full_text

%16 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%17 = ashr exact i64 %16, 32
%i648B

	full_text
	
i64 %16
5zext8B+
)
	full_text

%18 = zext i32 %3 to i64
0and8B'
%
	full_text

%19 = and i64 %18, 1
%i648B

	full_text
	
i64 %18
4icmp8B*
(
	full_text

%20 = icmp eq i32 %3, 1
:br8B2
0
	full_text#
!
br i1 %20, label %45, label %21
#i18B

	full_text


i1 %20
6sub8B-
+
	full_text

%22 = sub nsw i64 %18, %19
%i648B

	full_text
	
i64 %18
%i648B

	full_text
	
i64 %19
'br8B

	full_text

br label %23
Ophi8BF
D
	full_text7
5
3%24 = phi float [ 0.000000e+00, %21 ], [ %41, %23 ]
)float8B

	full_text

	float %41
Bphi8B9
7
	full_text*
(
&%25 = phi i64 [ 0, %21 ], [ %42, %23 ]
%i648B

	full_text
	
i64 %42
Dphi8B;
9
	full_text,
*
(%26 = phi i64 [ %22, %21 ], [ %43, %23 ]
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %43
6mul8B-
+
	full_text

%27 = mul nsw i64 %25, %15
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %15
6add8B-
+
	full_text

%28 = add nsw i64 %27, %17
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %17
\getelementptr8BI
G
	full_text:
8
6%29 = getelementptr inbounds float, float* %0, i64 %28
%i648B

	full_text
	
i64 %28
Lload8BB
@
	full_text3
1
/%30 = load float, float* %29, align 4, !tbaa !9
+float*8B

	full_text


float* %29
\getelementptr8BI
G
	full_text:
8
6%31 = getelementptr inbounds float, float* %1, i64 %25
%i648B

	full_text
	
i64 %25
Lload8BB
@
	full_text3
1
/%32 = load float, float* %31, align 4, !tbaa !9
+float*8B

	full_text


float* %31
ecall8B[
Y
	full_textL
J
H%33 = tail call float @llvm.fmuladd.f32(float %30, float %32, float %24)
)float8B

	full_text

	float %30
)float8B

	full_text

	float %32
)float8B

	full_text

	float %24
Lstore8BA
?
	full_text2
0
.store float %33, float* %12, align 4, !tbaa !9
)float8B

	full_text

	float %33
+float*8B

	full_text


float* %12
.or8B&
$
	full_text

%34 = or i64 %25, 1
%i648B

	full_text
	
i64 %25
6mul8B-
+
	full_text

%35 = mul nsw i64 %34, %15
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %15
6add8B-
+
	full_text

%36 = add nsw i64 %35, %17
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %17
\getelementptr8BI
G
	full_text:
8
6%37 = getelementptr inbounds float, float* %0, i64 %36
%i648B

	full_text
	
i64 %36
Lload8BB
@
	full_text3
1
/%38 = load float, float* %37, align 4, !tbaa !9
+float*8B

	full_text


float* %37
\getelementptr8BI
G
	full_text:
8
6%39 = getelementptr inbounds float, float* %1, i64 %34
%i648B

	full_text
	
i64 %34
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
H%41 = tail call float @llvm.fmuladd.f32(float %38, float %40, float %33)
)float8B

	full_text

	float %38
)float8B

	full_text

	float %40
)float8B

	full_text

	float %33
Lstore8BA
?
	full_text2
0
.store float %41, float* %12, align 4, !tbaa !9
)float8B

	full_text

	float %41
+float*8B

	full_text


float* %12
4add8B+
)
	full_text

%42 = add nsw i64 %25, 2
%i648B

	full_text
	
i64 %25
1add8B(
&
	full_text

%43 = add i64 %26, -2
%i648B

	full_text
	
i64 %26
5icmp8B+
)
	full_text

%44 = icmp eq i64 %43, 0
%i648B

	full_text
	
i64 %43
:br8B2
0
	full_text#
!
br i1 %44, label %45, label %23
#i18B

	full_text


i1 %44
Ophi8BF
D
	full_text7
5
3%46 = phi float [ 0.000000e+00, %14 ], [ %41, %23 ]
)float8B

	full_text

	float %41
Bphi8B9
7
	full_text*
(
&%47 = phi i64 [ 0, %14 ], [ %42, %23 ]
%i648B

	full_text
	
i64 %42
5icmp8B+
)
	full_text

%48 = icmp eq i64 %19, 0
%i648B

	full_text
	
i64 %19
:br8B2
0
	full_text#
!
br i1 %48, label %57, label %49
#i18B

	full_text


i1 %48
6mul8B-
+
	full_text

%50 = mul nsw i64 %47, %15
%i648B

	full_text
	
i64 %47
%i648B

	full_text
	
i64 %15
6add8B-
+
	full_text

%51 = add nsw i64 %50, %17
%i648B

	full_text
	
i64 %50
%i648B

	full_text
	
i64 %17
\getelementptr8BI
G
	full_text:
8
6%52 = getelementptr inbounds float, float* %0, i64 %51
%i648B

	full_text
	
i64 %51
Lload8BB
@
	full_text3
1
/%53 = load float, float* %52, align 4, !tbaa !9
+float*8B

	full_text


float* %52
\getelementptr8BI
G
	full_text:
8
6%54 = getelementptr inbounds float, float* %1, i64 %47
%i648B

	full_text
	
i64 %47
Lload8BB
@
	full_text3
1
/%55 = load float, float* %54, align 4, !tbaa !9
+float*8B

	full_text


float* %54
ecall8B[
Y
	full_textL
J
H%56 = tail call float @llvm.fmuladd.f32(float %53, float %55, float %46)
)float8B

	full_text

	float %53
)float8B

	full_text

	float %55
)float8B

	full_text

	float %46
Lstore8BA
?
	full_text2
0
.store float %56, float* %12, align 4, !tbaa !9
)float8B

	full_text

	float %56
+float*8B

	full_text


float* %12
'br8B

	full_text

br label %57
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %4
*float*8B

	full_text

	float* %2
*float*8B

	full_text

	float* %0
*float*8B
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
#i648B

	full_text	

i64 0
2float8B%
#
	full_text

float 0.000000e+00
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 1
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
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 -2      	  
 

                     !# "" $% $$ &' &( && )* )+ )) ,- ,. ,, /0 // 12 11 34 33 56 55 78 79 7: 77 ;< ;= ;; >? >> @A @B @@ CD CE CC FG FF HI HH JK JJ LM LL NO NP NQ NN RS RT RR UV UU WX WW YZ YY [\ [^ ]] _` __ ab aa cd cf eg ee hi hj hh kl kk mn mm op oo qr qq st su sv ss wx wy ww z| | | } } ~  / F k? 3? J? o    	 
          N #U % 'W ($ * +) - ., 0/ 2$ 43 61 85 9" :7 < =$ ?> A B@ D EC GF I> KJ MH OL P7 QN S T$ V& XW ZY \N ^U ` ba d_ f ge i jh lk n_ po rm tq u] vs x y  {  { ] c {c e! "z {[ ][ " ?? ?? {s ?? s ?? 7 ?? 7N ?? N? $	? Y? _	? a? ? "? ]	? 	? 	? >	? 	? 
	? 	? 	? U? 	? 	? W"
bicgKernel2"
_Z13get_global_idj"
llvm.fmuladd.f32*?
%polybench-gpu-1.0-bicg-bicgKernel2.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

wgsize
?

transfer_bytes
??? 

devmap_label

 
transfer_bytes_log1p
?.?A

wgsize_log1p
?.?A