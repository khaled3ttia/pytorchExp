
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
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 1) #3
4truncB+
)
	full_text

%9 = trunc i64 %8 to i32
"i64B

	full_text


i64 %8
2addB+
)
	full_text

%10 = add nsw i32 %3, -1
/subB(
&
	full_text

%11 = sub i32 %10, %4
#i32B

	full_text
	
i32 %10
5icmpB-
+
	full_text

%12 = icmp sgt i32 %11, %7
#i32B

	full_text
	
i32 %11
"i32B

	full_text


i32 %7
2subB+
)
	full_text

%13 = sub nsw i32 %3, %4
5icmpB-
+
	full_text

%14 = icmp sgt i32 %13, %9
#i32B

	full_text
	
i32 %13
"i32B

	full_text


i32 %9
/andB(
&
	full_text

%15 = and i1 %12, %14
!i1B

	full_text


i1 %12
!i1B

	full_text


i1 %14
8brB2
0
	full_text#
!
br i1 %15, label %16, label %48
!i1B

	full_text


i1 %15
/add8B&
$
	full_text

%17 = add i32 %4, 1
1add8B(
&
	full_text

%18 = add i32 %17, %7
%i328B

	full_text
	
i32 %17
$i328B

	full_text


i32 %7
5mul8B,
*
	full_text

%19 = mul nsw i32 %18, %3
%i328B

	full_text
	
i32 %18
5add8B,
*
	full_text

%20 = add nsw i32 %19, %4
%i328B

	full_text
	
i32 %19
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
6%22 = getelementptr inbounds float, float* %0, i64 %21
%i648B

	full_text
	
i64 %21
Lload8BB
@
	full_text3
1
/%23 = load float, float* %22, align 4, !tbaa !8
+float*8B

	full_text


float* %22
4mul8B+
)
	full_text

%24 = mul nsw i32 %4, %3
4add8B+
)
	full_text

%25 = add nsw i32 %9, %4
$i328B

	full_text


i32 %9
6add8B-
+
	full_text

%26 = add nsw i32 %25, %24
%i328B

	full_text
	
i32 %25
%i328B

	full_text
	
i32 %24
6sext8B,
*
	full_text

%27 = sext i32 %26 to i64
%i328B

	full_text
	
i32 %26
\getelementptr8BI
G
	full_text:
8
6%28 = getelementptr inbounds float, float* %1, i64 %27
%i648B

	full_text
	
i64 %27
Lload8BB
@
	full_text3
1
/%29 = load float, float* %28, align 4, !tbaa !8
+float*8B

	full_text


float* %28
6add8B-
+
	full_text

%30 = add nsw i32 %19, %25
%i328B

	full_text
	
i32 %19
%i328B

	full_text
	
i32 %25
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
6%32 = getelementptr inbounds float, float* %1, i64 %31
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
@fsub8B6
4
	full_text'
%
#%34 = fsub float -0.000000e+00, %23
)float8B

	full_text

	float %23
ecall8B[
Y
	full_textL
J
H%35 = tail call float @llvm.fmuladd.f32(float %34, float %29, float %33)
)float8B

	full_text

	float %34
)float8B

	full_text

	float %29
)float8B

	full_text

	float %33
Lstore8BA
?
	full_text2
0
.store float %35, float* %32, align 4, !tbaa !8
)float8B

	full_text

	float %35
+float*8B

	full_text


float* %32
4icmp8B*
(
	full_text

%36 = icmp eq i32 %9, 0
$i328B

	full_text


i32 %9
:br8B2
0
	full_text#
!
br i1 %36, label %37, label %48
#i18B

	full_text


i1 %36
\getelementptr8BI
G
	full_text:
8
6%38 = getelementptr inbounds float, float* %0, i64 %31
%i648B

	full_text
	
i64 %31
Lload8BB
@
	full_text3
1
/%39 = load float, float* %38, align 4, !tbaa !8
+float*8B

	full_text


float* %38
5sext8B+
)
	full_text

%40 = sext i32 %4 to i64
\getelementptr8BI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %2, i64 %40
%i648B

	full_text
	
i64 %40
Lload8BB
@
	full_text3
1
/%42 = load float, float* %41, align 4, !tbaa !8
+float*8B

	full_text


float* %41
6sext8B,
*
	full_text

%43 = sext i32 %18 to i64
%i328B

	full_text
	
i32 %18
\getelementptr8BI
G
	full_text:
8
6%44 = getelementptr inbounds float, float* %2, i64 %43
%i648B

	full_text
	
i64 %43
Lload8BB
@
	full_text3
1
/%45 = load float, float* %44, align 4, !tbaa !8
+float*8B

	full_text


float* %44
@fsub8B6
4
	full_text'
%
#%46 = fsub float -0.000000e+00, %39
)float8B

	full_text

	float %39
ecall8B[
Y
	full_textL
J
H%47 = tail call float @llvm.fmuladd.f32(float %46, float %42, float %45)
)float8B

	full_text

	float %46
)float8B

	full_text

	float %42
)float8B

	full_text

	float %45
Lstore8BA
?
	full_text2
0
.store float %47, float* %44, align 4, !tbaa !8
)float8B

	full_text

	float %47
+float*8B

	full_text


float* %44
'br8B

	full_text

br label %48
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %4
*float*8B

	full_text

	float* %2
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %3
*float*8B
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
$i328B

	full_text


i32 -1
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1
3float8B&
$
	full_text

float -0.000000e+00       	  
 
 

                    !    "# "" $$ %& %% '( ') '' *+ ** ,- ,, ./ .. 01 02 00 34 33 56 55 78 77 9: 99 ;< ;= ;> ;; ?@ ?A ?? BC BB DE DG FF HI HH JJ KL KK MN MM OP OO QR QQ ST SS UV UU WX WY WZ WW [\ [] [[ ^` ` ` ` ` $` %` Ja Ka Qb  b Fc c c c $d ,d 5   	    
         !  # &% ($ )' +* -, / 1% 20 43 65 8" :9 <. =7 >; @5 A CB E3 GF IJ LK N PO RQ TH VU XM YS ZW \Q ]  _D FD _^ _ _ ff ee ee ; ff ;W ff W ee g h h Bi i j 9j U"
Fan2"
_Z13get_global_idj"
llvm.fmuladd.f32*?
rodinia-3.1-gaussian-Fan2.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282

devmap_label
 

wgsize
 

transfer_bytes
?? 
 
transfer_bytes_log1p
|?RA

wgsize_log1p
|?RA