
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
3icmpB+
)
	full_text

%7 = icmp slt i32 %6, %3
"i32B

	full_text


i32 %6
2icmpB*
(
	full_text

%8 = icmp sgt i32 %3, 0
,andB%
#
	full_text

%9 = and i1 %7, %8
 i1B

	full_text	

i1 %7
 i1B

	full_text	

i1 %8
7brB1
/
	full_text"
 
br i1 %9, label %10, label %57
 i1B

	full_text	

i1 %9
0shl8B'
%
	full_text

%11 = shl i64 %5, 32
$i648B

	full_text


i64 %5
9ashr8B/
-
	full_text 

%12 = ashr exact i64 %11, 32
%i648B

	full_text
	
i64 %11
\getelementptr8BI
G
	full_text:
8
6%13 = getelementptr inbounds float, float* %1, i64 %12
%i648B

	full_text
	
i64 %12
5sext8B+
)
	full_text

%14 = sext i32 %3 to i64
0shl8B'
%
	full_text

%15 = shl i64 %5, 32
$i648B

	full_text


i64 %5
9ashr8B/
-
	full_text 

%16 = ashr exact i64 %15, 32
%i648B

	full_text
	
i64 %15
Lload8BB
@
	full_text3
1
/%17 = load float, float* %13, align 4, !tbaa !9
+float*8B

	full_text


float* %13
5zext8B+
)
	full_text

%18 = zext i32 %3 to i64
0and8B'
%
	full_text

%19 = and i64 %18, 1
%i648B

	full_text
	
i64 %18
4icmp8B*
(
	full_text

%20 = icmp eq i32 %3, 1
:br8B2
0
	full_text#
!
br i1 %20, label %45, label %21
#i18B

	full_text


i1 %20
6sub8B-
+
	full_text

%22 = sub nsw i64 %18, %19
%i648B

	full_text
	
i64 %18
%i648B

	full_text
	
i64 %19
'br8B

	full_text

br label %23
Fphi8B=
;
	full_text.
,
*%24 = phi float [ %17, %21 ], [ %41, %23 ]
)float8B

	full_text

	float %17
)float8B

	full_text

	float %41
Bphi8B9
7
	full_text*
(
&%25 = phi i64 [ 0, %21 ], [ %42, %23 ]
%i648B

	full_text
	
i64 %42
Dphi8B;
9
	full_text,
*
(%26 = phi i64 [ %22, %21 ], [ %43, %23 ]
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %43
6mul8B-
+
	full_text

%27 = mul nsw i64 %25, %14
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %14
6add8B-
+
	full_text

%28 = add nsw i64 %27, %16
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%29 = getelementptr inbounds float, float* %0, i64 %28
%i648B

	full_text
	
i64 %28
Lload8BB
@
	full_text3
1
/%30 = load float, float* %29, align 4, !tbaa !9
+float*8B

	full_text


float* %29
\getelementptr8BI
G
	full_text:
8
6%31 = getelementptr inbounds float, float* %2, i64 %25
%i648B

	full_text
	
i64 %25
Lload8BB
@
	full_text3
1
/%32 = load float, float* %31, align 4, !tbaa !9
+float*8B

	full_text


float* %31
ecall8B[
Y
	full_textL
J
H%33 = tail call float @llvm.fmuladd.f32(float %30, float %32, float %24)
)float8B

	full_text

	float %30
)float8B

	full_text

	float %32
)float8B

	full_text

	float %24
Lstore8BA
?
	full_text2
0
.store float %33, float* %13, align 4, !tbaa !9
)float8B

	full_text

	float %33
+float*8B

	full_text


float* %13
.or8B&
$
	full_text

%34 = or i64 %25, 1
%i648B

	full_text
	
i64 %25
6mul8B-
+
	full_text

%35 = mul nsw i64 %34, %14
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %14
6add8B-
+
	full_text

%36 = add nsw i64 %35, %16
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%37 = getelementptr inbounds float, float* %0, i64 %36
%i648B

	full_text
	
i64 %36
Lload8BB
@
	full_text3
1
/%38 = load float, float* %37, align 4, !tbaa !9
+float*8B

	full_text


float* %37
\getelementptr8BI
G
	full_text:
8
6%39 = getelementptr inbounds float, float* %2, i64 %34
%i648B

	full_text
	
i64 %34
Lload8BB
@
	full_text3
1
/%40 = load float, float* %39, align 4, !tbaa !9
+float*8B

	full_text


float* %39
ecall8B[
Y
	full_textL
J
H%41 = tail call float @llvm.fmuladd.f32(float %38, float %40, float %33)
)float8B

	full_text

	float %38
)float8B

	full_text

	float %40
)float8B

	full_text

	float %33
Lstore8BA
?
	full_text2
0
.store float %41, float* %13, align 4, !tbaa !9
)float8B

	full_text

	float %41
+float*8B

	full_text


float* %13
4add8B+
)
	full_text

%42 = add nsw i64 %25, 2
%i648B

	full_text
	
i64 %25
1add8B(
&
	full_text

%43 = add i64 %26, -2
%i648B

	full_text
	
i64 %26
5icmp8B+
)
	full_text

%44 = icmp eq i64 %43, 0
%i648B

	full_text
	
i64 %43
:br8B2
0
	full_text#
!
br i1 %44, label %45, label %23
#i18B

	full_text


i1 %44
Fphi8B=
;
	full_text.
,
*%46 = phi float [ %17, %10 ], [ %41, %23 ]
)float8B

	full_text

	float %17
)float8B

	full_text

	float %41
Bphi8B9
7
	full_text*
(
&%47 = phi i64 [ 0, %10 ], [ %42, %23 ]
%i648B

	full_text
	
i64 %42
5icmp8B+
)
	full_text

%48 = icmp eq i64 %19, 0
%i648B

	full_text
	
i64 %19
:br8B2
0
	full_text#
!
br i1 %48, label %57, label %49
#i18B

	full_text


i1 %48
6mul8B-
+
	full_text

%50 = mul nsw i64 %47, %14
%i648B

	full_text
	
i64 %47
%i648B

	full_text
	
i64 %14
6add8B-
+
	full_text

%51 = add nsw i64 %50, %16
%i648B

	full_text
	
i64 %50
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%52 = getelementptr inbounds float, float* %0, i64 %51
%i648B

	full_text
	
i64 %51
Lload8BB
@
	full_text3
1
/%53 = load float, float* %52, align 4, !tbaa !9
+float*8B

	full_text


float* %52
\getelementptr8BI
G
	full_text:
8
6%54 = getelementptr inbounds float, float* %2, i64 %47
%i648B

	full_text
	
i64 %47
Lload8BB
@
	full_text3
1
/%55 = load float, float* %54, align 4, !tbaa !9
+float*8B

	full_text


float* %54
ecall8B[
Y
	full_textL
J
H%56 = tail call float @llvm.fmuladd.f32(float %53, float %55, float %46)
)float8B

	full_text

	float %53
)float8B

	full_text

	float %55
)float8B

	full_text

	float %46
Lstore8BA
?
	full_text2
0
.store float %56, float* %13, align 4, !tbaa !9
)float8B

	full_text

	float %56
+float*8B

	full_text


float* %13
'br8B

	full_text

br label %57
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %0
*float*8B
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
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 -2
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 0
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 2       	  
 
                    !  "$ #% ## &' && () (* (( +, +- ++ ./ .0 .. 12 11 34 33 56 55 78 77 9: 9; 9< 99 => =? == @A @@ BC BD BB EF EG EE HI HH JK JJ LM LL NO NN PQ PR PS PP TU TV TT WX WW YZ YY [\ [[ ]^ ]` _a __ bc bb de dd fg fi hj hh kl km kk no nn pq pp rs rr tu tt vw vx vy vv z{ z| zz }     ? ? 1? H? n? 5? L? r    	            ! $P %W ' )Y *& , -+ / 0. 21 4& 65 83 :7 ;# <9 > ?& A@ C DB F GE IH K@ ML OJ QN R9 SP U V& X( ZY \[ ^ `P aW c ed gb i jh l mk on qb sr up wt x_ yv { |
 
 ~ _ f ~f h" #} ~] _] # ?? ~ ?? ?? v ?? vP ?? P9 ?? 9	? 	? @	? 	? Y? 	? ? &	? [? b	? d	? 	? 	? 	? 	? W"
mvt_kernel2"
_Z13get_global_idj"
llvm.fmuladd.f32*?
$polybench-gpu-1.0-mvt-mvt_kernel2.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

transfer_bytes
???@
 
transfer_bytes_log1p
D??A

wgsize_log1p
D??A

devmap_label


wgsize
?