

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 0) #2
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
1addB*
(
	full_text

%8 = add nsw i32 %3, -1
-subB&
$
	full_text

%9 = sub i32 %8, %4
"i32B

	full_text


i32 %8
4icmpB,
*
	full_text

%10 = icmp sgt i32 %9, %7
"i32B

	full_text


i32 %9
"i32B

	full_text


i32 %7
8brB2
0
	full_text#
!
br i1 %10, label %11, label %28
!i1B

	full_text


i1 %10
/add8B&
$
	full_text

%12 = add i32 %4, 1
1add8B(
&
	full_text

%13 = add i32 %12, %7
%i328B

	full_text
	
i32 %12
$i328B

	full_text


i32 %7
5mul8B,
*
	full_text

%14 = mul nsw i32 %13, %3
%i328B

	full_text
	
i32 %13
6sext8B,
*
	full_text

%15 = sext i32 %14 to i64
%i328B

	full_text
	
i32 %14
\getelementptr8BI
G
	full_text:
8
6%16 = getelementptr inbounds float, float* %1, i64 %15
%i648B

	full_text
	
i64 %15
5sext8B+
)
	full_text

%17 = sext i32 %4 to i64
]getelementptr8BJ
H
	full_text;
9
7%18 = getelementptr inbounds float, float* %16, i64 %17
+float*8B

	full_text


float* %16
%i648B

	full_text
	
i64 %17
Lload8BB
@
	full_text3
1
/%19 = load float, float* %18, align 4, !tbaa !8
+float*8B

	full_text


float* %18
4mul8B+
)
	full_text

%20 = mul nsw i32 %4, %3
6sext8B,
*
	full_text

%21 = sext i32 %20 to i64
%i328B

	full_text
	
i32 %20
\getelementptr8BI
G
	full_text:
8
6%22 = getelementptr inbounds float, float* %1, i64 %21
%i648B

	full_text
	
i64 %21
]getelementptr8BJ
H
	full_text;
9
7%23 = getelementptr inbounds float, float* %22, i64 %17
+float*8B

	full_text


float* %22
%i648B

	full_text
	
i64 %17
Lload8BB
@
	full_text3
1
/%24 = load float, float* %23, align 4, !tbaa !8
+float*8B

	full_text


float* %23
Cfdiv8B9
7
	full_text*
(
&%25 = fdiv float %19, %24, !fpmath !12
)float8B

	full_text

	float %19
)float8B

	full_text

	float %24
\getelementptr8BI
G
	full_text:
8
6%26 = getelementptr inbounds float, float* %0, i64 %15
%i648B

	full_text
	
i64 %15
]getelementptr8BJ
H
	full_text;
9
7%27 = getelementptr inbounds float, float* %26, i64 %17
+float*8B

	full_text


float* %26
%i648B

	full_text
	
i64 %17
Lstore8BA
?
	full_text2
0
.store float %25, float* %27, align 4, !tbaa !8
)float8B

	full_text

	float %25
+float*8B

	full_text


float* %27
'br8B

	full_text

br label %28
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %4
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %0
*float*8B

	full_text

	float* %1
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 0
$i328B

	full_text


i32 -1       	  
 
                      !" !# !! $% $$ &' &( && )* )) +, +- ++ ./ .0 .. 13 3 3 3 4 4 4 5 )6 6     	             " #! % '$ ( *) , -& /+ 0
 
 21 2 2 77 77 8 9 : "
Fan1"
_Z13get_global_idj*?
rodinia-3.1-gaussian-Fan1.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02

wgsize_log1p
|?RA

wgsize
 

transfer_bytes
?? 

devmap_label
 
 
transfer_bytes_log1p
|?RA