
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
6brB0
.
	full_text!

br i1 %7, label %8, label %40
 i1B

	full_text	

i1 %7
.shl8B%
#
	full_text

%9 = shl i32 %6, 2
$i328B

	full_text


i32 %6
5sext8B+
)
	full_text

%10 = sext i32 %9 to i64
$i328B

	full_text


i32 %9
\getelementptr8BI
G
	full_text:
8
6%11 = getelementptr inbounds float, float* %0, i64 %10
%i648B

	full_text
	
i64 %10
Lload8BB
@
	full_text3
1
/%12 = load float, float* %11, align 4, !tbaa !8
+float*8B

	full_text


float* %11
\getelementptr8BI
G
	full_text:
8
6%13 = getelementptr inbounds float, float* %1, i64 %10
%i648B

	full_text
	
i64 %10
Lload8BB
@
	full_text3
1
/%14 = load float, float* %13, align 4, !tbaa !8
+float*8B

	full_text


float* %13
-or8B%
#
	full_text

%15 = or i32 %9, 1
$i328B

	full_text


i32 %9
6sext8B,
*
	full_text

%16 = sext i32 %15 to i64
%i328B

	full_text
	
i32 %15
\getelementptr8BI
G
	full_text:
8
6%17 = getelementptr inbounds float, float* %0, i64 %16
%i648B

	full_text
	
i64 %16
Lload8BB
@
	full_text3
1
/%18 = load float, float* %17, align 4, !tbaa !8
+float*8B

	full_text


float* %17
\getelementptr8BI
G
	full_text:
8
6%19 = getelementptr inbounds float, float* %1, i64 %16
%i648B

	full_text
	
i64 %16
Lload8BB
@
	full_text3
1
/%20 = load float, float* %19, align 4, !tbaa !8
+float*8B

	full_text


float* %19
6fmul8B,
*
	full_text

%21 = fmul float %18, %20
)float8B

	full_text

	float %18
)float8B

	full_text

	float %20
ecall8B[
Y
	full_textL
J
H%22 = tail call float @llvm.fmuladd.f32(float %12, float %14, float %21)
)float8B

	full_text

	float %12
)float8B

	full_text

	float %14
)float8B

	full_text

	float %21
-or8B%
#
	full_text

%23 = or i32 %9, 2
$i328B

	full_text


i32 %9
6sext8B,
*
	full_text

%24 = sext i32 %23 to i64
%i328B

	full_text
	
i32 %23
\getelementptr8BI
G
	full_text:
8
6%25 = getelementptr inbounds float, float* %0, i64 %24
%i648B

	full_text
	
i64 %24
Lload8BB
@
	full_text3
1
/%26 = load float, float* %25, align 4, !tbaa !8
+float*8B

	full_text


float* %25
\getelementptr8BI
G
	full_text:
8
6%27 = getelementptr inbounds float, float* %1, i64 %24
%i648B

	full_text
	
i64 %24
Lload8BB
@
	full_text3
1
/%28 = load float, float* %27, align 4, !tbaa !8
+float*8B

	full_text


float* %27
ecall8B[
Y
	full_textL
J
H%29 = tail call float @llvm.fmuladd.f32(float %26, float %28, float %22)
)float8B

	full_text

	float %26
)float8B

	full_text

	float %28
)float8B

	full_text

	float %22
-or8B%
#
	full_text

%30 = or i32 %9, 3
$i328B

	full_text


i32 %9
6sext8B,
*
	full_text

%31 = sext i32 %30 to i64
%i328B

	full_text
	
i32 %30
\getelementptr8BI
G
	full_text:
8
6%32 = getelementptr inbounds float, float* %0, i64 %31
%i648B

	full_text
	
i64 %31
Lload8BB
@
	full_text3
1
/%33 = load float, float* %32, align 4, !tbaa !8
+float*8B

	full_text


float* %32
\getelementptr8BI
G
	full_text:
8
6%34 = getelementptr inbounds float, float* %1, i64 %31
%i648B

	full_text
	
i64 %31
Lload8BB
@
	full_text3
1
/%35 = load float, float* %34, align 4, !tbaa !8
+float*8B

	full_text


float* %34
ecall8B[
Y
	full_textL
J
H%36 = tail call float @llvm.fmuladd.f32(float %33, float %35, float %29)
)float8B

	full_text

	float %33
)float8B

	full_text

	float %35
)float8B

	full_text

	float %29
0shl8B'
%
	full_text

%37 = shl i64 %5, 32
$i648B

	full_text


i64 %5
9ashr8B/
-
	full_text 

%38 = ashr exact i64 %37, 32
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%39 = getelementptr inbounds float, float* %2, i64 %38
%i648B

	full_text
	
i64 %38
Lstore8BA
?
	full_text2
0
.store float %36, float* %39, align 4, !tbaa !8
)float8B

	full_text

	float %36
+float*8B

	full_text


float* %39
'br8B

	full_text

br label %40
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %0
*float*8B

	full_text

	float* %2
*float*8B

	full_text

	float* %1
$i328B

	full_text


i32 %3
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
#i328B

	full_text	

i32 2
#i328B

	full_text	

i32 3
$i648B

	full_text


i64 32      	  
 

                      !  "    #$ #% #& ## '( '' )* )) +, ++ -. -- /0 // 12 11 34 35 36 33 78 77 9: 99 ;< ;; => == ?@ ?? AB AA CD CE CF CC GH GG IJ II KL KK MN MO MM PR R R +R ;S KT T T /T ?U     	 
  
         ! " $ %  & (' *) ,+ .) 0/ 2- 41 5# 6 87 :9 <; >9 @? B= DA E3 F HG JI LC NK O  QP Q Q VV WW# WW #3 WW 3 VV C WW CX Y Z Z '[ 7\ G\ I"

DotProduct"
_Z13get_global_idj"
llvm.fmuladd.f32*?
#nvidia-4.2-DotProduct-DotProduct.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

transfer_bytes
???

wgsize
?
 
transfer_bytes_log1p
?'?A

wgsize_log1p
?'?A

devmap_label
