

[external]
JcallBB
@
	full_text3
1
/%7 = tail call i64 @_Z12get_group_idj(i32 1) #4
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
JcallBB
@
	full_text3
1
/%9 = tail call i64 @_Z12get_local_idj(i32 0) #4
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
KcallBC
A
	full_text4
2
0%11 = tail call i64 @_Z12get_local_idj(i32 1) #4
6truncB-
+
	full_text

%12 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
1addB*
(
	full_text

%13 = add nsw i32 %1, 1
-shlB&
$
	full_text

%14 = shl i32 %8, 4
"i32B

	full_text


i32 %8
0addB)
'
	full_text

%15 = add i32 %14, %12
#i32B

	full_text
	
i32 %14
#i32B

	full_text
	
i32 %12
0mulB)
'
	full_text

%16 = mul i32 %15, %13
#i32B

	full_text
	
i32 %15
#i32B

	full_text
	
i32 %13
-addB&
$
	full_text

%17 = add i32 %1, 2
0addB)
'
	full_text

%18 = add i32 %17, %10
#i32B

	full_text
	
i32 %17
#i32B

	full_text
	
i32 %10
0addB)
'
	full_text

%19 = add i32 %18, %16
#i32B

	full_text
	
i32 %18
#i32B

	full_text
	
i32 %16
,orB&
$
	full_text

%20 = or i32 %14, 1
#i32B

	full_text
	
i32 %14
0addB)
'
	full_text

%21 = add i32 %20, %12
#i32B

	full_text
	
i32 %20
#i32B

	full_text
	
i32 %12
.shlB'
%
	full_text

%22 = shl i64 %9, 32
"i64B

	full_text


i64 %9
7addB0
.
	full_text!

%23 = add i64 %22, 4294967296
#i64B

	full_text
	
i64 %22
7ashrB/
-
	full_text 

%24 = ashr exact i64 %23, 32
#i64B

	full_text
	
i64 %23
ZgetelementptrBI
G
	full_text:
8
6%25 = getelementptr inbounds float, float* %0, i64 %24
#i64B

	full_text
	
i64 %24
JloadBB
@
	full_text3
1
/%26 = load float, float* %25, align 4, !tbaa !8
)float*B

	full_text


float* %25
CfmulB;
9
	full_text,
*
(%27 = fmul float %26, 0x3FD3333340000000
'floatB

	full_text

	float %26
4sextB,
*
	full_text

%28 = sext i32 %21 to i64
#i32B

	full_text
	
i32 %21
ZgetelementptrBI
G
	full_text:
8
6%29 = getelementptr inbounds float, float* %2, i64 %28
#i64B

	full_text
	
i64 %28
JloadBB
@
	full_text3
1
/%30 = load float, float* %29, align 4, !tbaa !8
)float*B

	full_text


float* %29
4sextB,
*
	full_text

%31 = sext i32 %19 to i64
#i32B

	full_text
	
i32 %19
ZgetelementptrBI
G
	full_text:
8
6%32 = getelementptr inbounds float, float* %5, i64 %31
#i64B

	full_text
	
i64 %31
JloadBB
@
	full_text3
1
/%33 = load float, float* %32, align 4, !tbaa !8
)float*B

	full_text


float* %32
CfmulB;
9
	full_text,
*
(%34 = fmul float %33, 0x3FD3333340000000
'floatB

	full_text

	float %33
ccallB[
Y
	full_textL
J
H%35 = tail call float @llvm.fmuladd.f32(float %27, float %30, float %34)
'floatB

	full_text

	float %27
'floatB

	full_text

	float %30
'floatB

	full_text

	float %34
ZgetelementptrBI
G
	full_text:
8
6%36 = getelementptr inbounds float, float* %4, i64 %31
#i64B

	full_text
	
i64 %31
JloadBB
@
	full_text3
1
/%37 = load float, float* %36, align 4, !tbaa !8
)float*B

	full_text


float* %36
4faddB,
*
	full_text

%38 = fadd float %37, %35
'floatB

	full_text

	float %37
'floatB

	full_text

	float %35
JstoreBA
?
	full_text2
0
.store float %38, float* %36, align 4, !tbaa !8
'floatB

	full_text

	float %38
)float*B

	full_text


float* %36
JloadBB
@
	full_text3
1
/%39 = load float, float* %25, align 4, !tbaa !8
)float*B

	full_text


float* %25
CfmulB;
9
	full_text,
*
(%40 = fmul float %39, 0x3FD3333340000000
'floatB

	full_text

	float %39
JloadBB
@
	full_text3
1
/%41 = load float, float* %29, align 4, !tbaa !8
)float*B

	full_text


float* %29
JloadBB
@
	full_text3
1
/%42 = load float, float* %32, align 4, !tbaa !8
)float*B

	full_text


float* %32
CfmulB;
9
	full_text,
*
(%43 = fmul float %42, 0x3FD3333340000000
'floatB

	full_text

	float %42
ccallB[
Y
	full_textL
J
H%44 = tail call float @llvm.fmuladd.f32(float %40, float %41, float %43)
'floatB

	full_text

	float %40
'floatB

	full_text

	float %41
'floatB

	full_text

	float %43
JstoreBA
?
	full_text2
0
.store float %44, float* %32, align 4, !tbaa !8
'floatB

	full_text

	float %44
)float*B

	full_text


float* %32
@callB8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
-orB'
%
	full_text

%45 = or i64 %11, %7
#i64B

	full_text
	
i64 %11
"i64B

	full_text


i64 %7
6truncB-
+
	full_text

%46 = trunc i64 %45 to i32
#i64B

	full_text
	
i64 %45
3icmpB+
)
	full_text

%47 = icmp eq i32 %46, 0
#i32B

	full_text
	
i32 %46
8brB2
0
	full_text#
!
br i1 %47, label %48, label %61
!i1B

	full_text


i1 %47
Lload8BB
@
	full_text3
1
/%49 = load float, float* %25, align 4, !tbaa !8
+float*8B

	full_text


float* %25
\getelementptr8BI
G
	full_text:
8
6%50 = getelementptr inbounds float, float* %5, i64 %24
%i648B

	full_text
	
i64 %24
Lload8BB
@
	full_text3
1
/%51 = load float, float* %50, align 4, !tbaa !8
+float*8B

	full_text


float* %50
Efmul8B;
9
	full_text,
*
(%52 = fmul float %51, 0x3FD3333340000000
)float8B

	full_text

	float %51
tcall8Bj
h
	full_text[
Y
W%53 = tail call float @llvm.fmuladd.f32(float %49, float 0x3FD3333340000000, float %52)
)float8B

	full_text

	float %49
)float8B

	full_text

	float %52
\getelementptr8BI
G
	full_text:
8
6%54 = getelementptr inbounds float, float* %4, i64 %24
%i648B

	full_text
	
i64 %24
Lload8BB
@
	full_text3
1
/%55 = load float, float* %54, align 4, !tbaa !8
+float*8B

	full_text


float* %54
6fadd8B,
*
	full_text

%56 = fadd float %55, %53
)float8B

	full_text

	float %55
)float8B

	full_text

	float %53
Lstore8BA
?
	full_text2
0
.store float %56, float* %54, align 4, !tbaa !8
)float8B

	full_text

	float %56
+float*8B

	full_text


float* %54
Lload8BB
@
	full_text3
1
/%57 = load float, float* %25, align 4, !tbaa !8
+float*8B

	full_text


float* %25
Lload8BB
@
	full_text3
1
/%58 = load float, float* %50, align 4, !tbaa !8
+float*8B

	full_text


float* %50
Efmul8B;
9
	full_text,
*
(%59 = fmul float %58, 0x3FD3333340000000
)float8B

	full_text

	float %58
tcall8Bj
h
	full_text[
Y
W%60 = tail call float @llvm.fmuladd.f32(float %57, float 0x3FD3333340000000, float %59)
)float8B

	full_text

	float %57
)float8B

	full_text

	float %59
Lstore8BA
?
	full_text2
0
.store float %60, float* %50, align 4, !tbaa !8
)float8B

	full_text

	float %60
+float*8B

	full_text


float* %50
'br8B

	full_text

br label %61
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %2
*float*8B

	full_text

	float* %4
*float*8B

	full_text

	float* %5
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %1
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
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 4
#i328B

	full_text	

i32 2
,i648B!

	full_text

i64 4294967296
8float8B+
)
	full_text

float 0x3FD3333340000000       	  

                        !" !! #$ ## %& %% '( '' )* )) +, ++ -. -- /0 // 12 11 34 33 56 55 78 77 9: 9; 9< 99 => == ?@ ?? AB AC AA DE DF DD GH GG IJ II KL KK MN MM OP OO QR QS QT QQ UV UW UU XX YZ Y[ YY \] \\ ^_ ^^ `a `c bb de dd fg ff hi hh jk jl jj mn mm op oo qr qs qq tu tv tt wx ww yz yy {| {{ }~ } }} ?? ?
? ?? ?? -? =? m? 3? d? %? 
?    	    
           "! $# &% (' * ,+ .- 0 21 43 65 8) :/ ;7 <1 >= @? B9 CA E= F% HG J- L3 NM PI RK SO TQ V3 W Z [Y ]\ _^ a% c# ed gf ib kh l# nm po rj sq um v% xd zy |w ~{ } ?d ?` b` ?? ? ?? ?? ? ?? ?? ??  ?? j ?? j ?? 9 ?? 9X ?? XQ ?? Q} ?? }? 	? ^? ? 	? 
	? ? X	? 	? #	? 	? 	? !	? )	? 7	? I	? O	? h	? j	? {	? }"
bpnn_adjust_weights_ocl"
_Z12get_group_idj"
_Z12get_local_idj"
llvm.fmuladd.f32"
_Z7barrierj*?
/rodinia-3.1-backprop-bpnn_adjust_weights_ocl.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?
 
transfer_bytes_log1p
I{?A

transfer_bytes
???

wgsize_log1p
I{?A

wgsize
?

devmap_label
 