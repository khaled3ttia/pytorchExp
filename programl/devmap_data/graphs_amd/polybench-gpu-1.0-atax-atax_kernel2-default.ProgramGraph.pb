
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
2icmpB*
(
	full_text

%9 = icmp sgt i32 %3, 0
-andB&
$
	full_text

%10 = and i1 %8, %9
 i1B

	full_text	

i1 %8
 i1B

	full_text	

i1 %9
8brB2
0
	full_text#
!
br i1 %10, label %11, label %58
!i1B

	full_text


i1 %10
0shl8B'
%
	full_text

%12 = shl i64 %6, 32
$i648B

	full_text


i64 %6
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
5sext8B+
)
	full_text

%15 = sext i32 %4 to i64
0shl8B'
%
	full_text

%16 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%17 = ashr exact i64 %16, 32
%i648B

	full_text
	
i64 %16
Lload8BB
@
	full_text3
1
/%18 = load float, float* %14, align 4, !tbaa !9
+float*8B

	full_text


float* %14
5zext8B+
)
	full_text

%19 = zext i32 %3 to i64
0and8B'
%
	full_text

%20 = and i64 %19, 1
%i648B

	full_text
	
i64 %19
4icmp8B*
(
	full_text

%21 = icmp eq i32 %3, 1
:br8B2
0
	full_text#
!
br i1 %21, label %46, label %22
#i18B

	full_text


i1 %21
6sub8B-
+
	full_text

%23 = sub nsw i64 %19, %20
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %20
'br8B

	full_text

br label %24
Fphi8B=
;
	full_text.
,
*%25 = phi float [ %18, %22 ], [ %42, %24 ]
)float8B

	full_text

	float %18
)float8B

	full_text

	float %42
Bphi8B9
7
	full_text*
(
&%26 = phi i64 [ 0, %22 ], [ %43, %24 ]
%i648B

	full_text
	
i64 %43
Dphi8B;
9
	full_text,
*
(%27 = phi i64 [ %23, %22 ], [ %44, %24 ]
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %44
6mul8B-
+
	full_text

%28 = mul nsw i64 %26, %15
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %15
6add8B-
+
	full_text

%29 = add nsw i64 %28, %17
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %17
\getelementptr8BI
G
	full_text:
8
6%30 = getelementptr inbounds float, float* %0, i64 %29
%i648B

	full_text
	
i64 %29
Lload8BB
@
	full_text3
1
/%31 = load float, float* %30, align 4, !tbaa !9
+float*8B

	full_text


float* %30
\getelementptr8BI
G
	full_text:
8
6%32 = getelementptr inbounds float, float* %2, i64 %26
%i648B

	full_text
	
i64 %26
Lload8BB
@
	full_text3
1
/%33 = load float, float* %32, align 4, !tbaa !9
+float*8B

	full_text


float* %32
ecall8B[
Y
	full_textL
J
H%34 = tail call float @llvm.fmuladd.f32(float %31, float %33, float %25)
)float8B

	full_text

	float %31
)float8B

	full_text

	float %33
)float8B

	full_text

	float %25
Lstore8BA
?
	full_text2
0
.store float %34, float* %14, align 4, !tbaa !9
)float8B

	full_text

	float %34
+float*8B

	full_text


float* %14
.or8B&
$
	full_text

%35 = or i64 %26, 1
%i648B

	full_text
	
i64 %26
6mul8B-
+
	full_text

%36 = mul nsw i64 %35, %15
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %15
6add8B-
+
	full_text

%37 = add nsw i64 %36, %17
%i648B

	full_text
	
i64 %36
%i648B

	full_text
	
i64 %17
\getelementptr8BI
G
	full_text:
8
6%38 = getelementptr inbounds float, float* %0, i64 %37
%i648B

	full_text
	
i64 %37
Lload8BB
@
	full_text3
1
/%39 = load float, float* %38, align 4, !tbaa !9
+float*8B

	full_text


float* %38
\getelementptr8BI
G
	full_text:
8
6%40 = getelementptr inbounds float, float* %2, i64 %35
%i648B

	full_text
	
i64 %35
Lload8BB
@
	full_text3
1
/%41 = load float, float* %40, align 4, !tbaa !9
+float*8B

	full_text


float* %40
ecall8B[
Y
	full_textL
J
H%42 = tail call float @llvm.fmuladd.f32(float %39, float %41, float %34)
)float8B

	full_text

	float %39
)float8B

	full_text

	float %41
)float8B

	full_text

	float %34
Lstore8BA
?
	full_text2
0
.store float %42, float* %14, align 4, !tbaa !9
)float8B

	full_text

	float %42
+float*8B

	full_text


float* %14
4add8B+
)
	full_text

%43 = add nsw i64 %26, 2
%i648B

	full_text
	
i64 %26
1add8B(
&
	full_text

%44 = add i64 %27, -2
%i648B

	full_text
	
i64 %27
5icmp8B+
)
	full_text

%45 = icmp eq i64 %44, 0
%i648B

	full_text
	
i64 %44
:br8B2
0
	full_text#
!
br i1 %45, label %46, label %24
#i18B

	full_text


i1 %45
Fphi8B=
;
	full_text.
,
*%47 = phi float [ %18, %11 ], [ %42, %24 ]
)float8B

	full_text

	float %18
)float8B

	full_text

	float %42
Bphi8B9
7
	full_text*
(
&%48 = phi i64 [ 0, %11 ], [ %43, %24 ]
%i648B

	full_text
	
i64 %43
5icmp8B+
)
	full_text

%49 = icmp eq i64 %20, 0
%i648B

	full_text
	
i64 %20
:br8B2
0
	full_text#
!
br i1 %49, label %58, label %50
#i18B

	full_text


i1 %49
6mul8B-
+
	full_text

%51 = mul nsw i64 %48, %15
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %15
6add8B-
+
	full_text

%52 = add nsw i64 %51, %17
%i648B

	full_text
	
i64 %51
%i648B

	full_text
	
i64 %17
\getelementptr8BI
G
	full_text:
8
6%53 = getelementptr inbounds float, float* %0, i64 %52
%i648B

	full_text
	
i64 %52
Lload8BB
@
	full_text3
1
/%54 = load float, float* %53, align 4, !tbaa !9
+float*8B

	full_text


float* %53
\getelementptr8BI
G
	full_text:
8
6%55 = getelementptr inbounds float, float* %2, i64 %48
%i648B

	full_text
	
i64 %48
Lload8BB
@
	full_text3
1
/%56 = load float, float* %55, align 4, !tbaa !9
+float*8B

	full_text


float* %55
ecall8B[
Y
	full_textL
J
H%57 = tail call float @llvm.fmuladd.f32(float %54, float %56, float %47)
)float8B

	full_text

	float %54
)float8B

	full_text

	float %56
)float8B

	full_text

	float %47
Lstore8BA
?
	full_text2
0
.store float %57, float* %14, align 4, !tbaa !9
)float8B

	full_text

	float %57
+float*8B

	full_text


float* %14
'br8B

	full_text

br label %58
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %2
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %1
$i328B

	full_text


i32 %4
*float*8B
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
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 2
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 0
$i648B

	full_text


i64 -2
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 0       	  
 
                    !  "$ #% ## &' && () (* (( +, +- ++ ./ .0 .. 12 11 34 33 56 55 78 77 9: 9; 9< 99 => =? == @A @@ BC BD BB EF EG EE HI HH JK JJ LM LL NO NN PQ PR PS PP TU TV TT WX WW YZ YY [\ [[ ]^ ]` _a __ bc bb de dd fg fi hj hh kl km kk no nn pq pp rs rr tu tt vw vx vy vv z{ z| zz } 5 L r? ? ? ? 	? ? ? 1? H? n    	            ! $P %W ' )Y *& , -+ / 0. 21 4& 65 83 :7 ;# <9 > ?& A@ C DB F GE IH K@ ML OJ QN R9 SP U V& X( ZY \[ ^ `P aW c ed gb i jh l mk on qb sr up wt x_ yv { |
 
 ~ _ f ~f h" #} ~] _] # ~ ?? ??P ?? Pv ?? v9 ?? 9 ?? 	? 	? W	? 	? @? &	? [? b	? d	? Y	? 	? 	? 	? ? 	? "
atax_kernel2"
_Z13get_global_idj"
llvm.fmuladd.f32*?
&polybench-gpu-1.0-atax-atax_kernel2.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

transfer_bytes
??? 

devmap_label

 
transfer_bytes_log1p
3.?A

wgsize_log1p
3.?A

wgsize
?