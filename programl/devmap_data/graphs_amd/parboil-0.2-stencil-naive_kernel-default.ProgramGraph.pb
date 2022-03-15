

[external]
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 0) #3
,addB%
#
	full_text

%9 = add i64 %8, 1
"i64B

	full_text


i64 %8
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
LcallBD
B
	full_text5
3
1%11 = tail call i64 @_Z13get_global_idj(i32 1) #3
.addB'
%
	full_text

%12 = add i64 %11, 1
#i64B

	full_text
	
i64 %11
6truncB-
+
	full_text

%13 = trunc i64 %12 to i32
#i64B

	full_text
	
i64 %12
LcallBD
B
	full_text5
3
1%14 = tail call i64 @_Z13get_global_idj(i32 2) #3
.addB'
%
	full_text

%15 = add i64 %14, 1
#i64B

	full_text
	
i64 %14
6truncB-
+
	full_text

%16 = trunc i64 %15 to i32
#i64B

	full_text
	
i64 %15
2addB+
)
	full_text

%17 = add nsw i32 %4, -1
6icmpB.
,
	full_text

%18 = icmp sgt i32 %17, %10
#i32B

	full_text
	
i32 %17
#i32B

	full_text
	
i32 %10
8brB2
0
	full_text#
!
br i1 %18, label %19, label %76
!i1B

	full_text


i1 %18
4add8B+
)
	full_text

%20 = add nsw i32 %16, 1
%i328B

	full_text
	
i32 %16
5mul8B,
*
	full_text

%21 = mul nsw i32 %20, %5
%i328B

	full_text
	
i32 %20
6add8B-
+
	full_text

%22 = add nsw i32 %21, %13
%i328B

	full_text
	
i32 %21
%i328B

	full_text
	
i32 %13
5mul8B,
*
	full_text

%23 = mul nsw i32 %22, %4
%i328B

	full_text
	
i32 %22
6add8B-
+
	full_text

%24 = add nsw i32 %23, %10
%i328B

	full_text
	
i32 %23
%i328B

	full_text
	
i32 %10
6sext8B,
*
	full_text

%25 = sext i32 %24 to i64
%i328B

	full_text
	
i32 %24
\getelementptr8BI
G
	full_text:
8
6%26 = getelementptr inbounds float, float* %2, i64 %25
%i648B

	full_text
	
i64 %25
Lload8BB
@
	full_text3
1
/%27 = load float, float* %26, align 4, !tbaa !8
+float*8B

	full_text


float* %26
5add8B,
*
	full_text

%28 = add nsw i32 %16, -1
%i328B

	full_text
	
i32 %16
5mul8B,
*
	full_text

%29 = mul nsw i32 %28, %5
%i328B

	full_text
	
i32 %28
6add8B-
+
	full_text

%30 = add nsw i32 %29, %13
%i328B

	full_text
	
i32 %29
%i328B

	full_text
	
i32 %13
5mul8B,
*
	full_text

%31 = mul nsw i32 %30, %4
%i328B

	full_text
	
i32 %30
6add8B-
+
	full_text

%32 = add nsw i32 %31, %10
%i328B

	full_text
	
i32 %31
%i328B

	full_text
	
i32 %10
6sext8B,
*
	full_text

%33 = sext i32 %32 to i64
%i328B

	full_text
	
i32 %32
\getelementptr8BI
G
	full_text:
8
6%34 = getelementptr inbounds float, float* %2, i64 %33
%i648B

	full_text
	
i64 %33
Lload8BB
@
	full_text3
1
/%35 = load float, float* %34, align 4, !tbaa !8
+float*8B

	full_text


float* %34
6fadd8B,
*
	full_text

%36 = fadd float %27, %35
)float8B

	full_text

	float %27
)float8B

	full_text

	float %35
4add8B+
)
	full_text

%37 = add nsw i32 %13, 1
%i328B

	full_text
	
i32 %13
5mul8B,
*
	full_text

%38 = mul nsw i32 %16, %5
%i328B

	full_text
	
i32 %16
6add8B-
+
	full_text

%39 = add nsw i32 %37, %38
%i328B

	full_text
	
i32 %37
%i328B

	full_text
	
i32 %38
5mul8B,
*
	full_text

%40 = mul nsw i32 %39, %4
%i328B

	full_text
	
i32 %39
6add8B-
+
	full_text

%41 = add nsw i32 %40, %10
%i328B

	full_text
	
i32 %40
%i328B

	full_text
	
i32 %10
6sext8B,
*
	full_text

%42 = sext i32 %41 to i64
%i328B

	full_text
	
i32 %41
\getelementptr8BI
G
	full_text:
8
6%43 = getelementptr inbounds float, float* %2, i64 %42
%i648B

	full_text
	
i64 %42
Lload8BB
@
	full_text3
1
/%44 = load float, float* %43, align 4, !tbaa !8
+float*8B

	full_text


float* %43
6fadd8B,
*
	full_text

%45 = fadd float %36, %44
)float8B

	full_text

	float %36
)float8B

	full_text

	float %44
5add8B,
*
	full_text

%46 = add nsw i32 %13, -1
%i328B

	full_text
	
i32 %13
6add8B-
+
	full_text

%47 = add nsw i32 %46, %38
%i328B

	full_text
	
i32 %46
%i328B

	full_text
	
i32 %38
5mul8B,
*
	full_text

%48 = mul nsw i32 %47, %4
%i328B

	full_text
	
i32 %47
6add8B-
+
	full_text

%49 = add nsw i32 %48, %10
%i328B

	full_text
	
i32 %48
%i328B

	full_text
	
i32 %10
6sext8B,
*
	full_text

%50 = sext i32 %49 to i64
%i328B

	full_text
	
i32 %49
\getelementptr8BI
G
	full_text:
8
6%51 = getelementptr inbounds float, float* %2, i64 %50
%i648B

	full_text
	
i64 %50
Lload8BB
@
	full_text3
1
/%52 = load float, float* %51, align 4, !tbaa !8
+float*8B

	full_text


float* %51
6fadd8B,
*
	full_text

%53 = fadd float %45, %52
)float8B

	full_text

	float %45
)float8B

	full_text

	float %52
4add8B+
)
	full_text

%54 = add nsw i32 %10, 1
%i328B

	full_text
	
i32 %10
6add8B-
+
	full_text

%55 = add nsw i32 %38, %13
%i328B

	full_text
	
i32 %38
%i328B

	full_text
	
i32 %13
5mul8B,
*
	full_text

%56 = mul nsw i32 %55, %4
%i328B

	full_text
	
i32 %55
6add8B-
+
	full_text

%57 = add nsw i32 %54, %56
%i328B

	full_text
	
i32 %54
%i328B

	full_text
	
i32 %56
6sext8B,
*
	full_text

%58 = sext i32 %57 to i64
%i328B

	full_text
	
i32 %57
\getelementptr8BI
G
	full_text:
8
6%59 = getelementptr inbounds float, float* %2, i64 %58
%i648B

	full_text
	
i64 %58
Lload8BB
@
	full_text3
1
/%60 = load float, float* %59, align 4, !tbaa !8
+float*8B

	full_text


float* %59
6fadd8B,
*
	full_text

%61 = fadd float %53, %60
)float8B

	full_text

	float %53
)float8B

	full_text

	float %60
5add8B,
*
	full_text

%62 = add nsw i32 %10, -1
%i328B

	full_text
	
i32 %10
6add8B-
+
	full_text

%63 = add nsw i32 %62, %56
%i328B

	full_text
	
i32 %62
%i328B

	full_text
	
i32 %56
6sext8B,
*
	full_text

%64 = sext i32 %63 to i64
%i328B

	full_text
	
i32 %63
\getelementptr8BI
G
	full_text:
8
6%65 = getelementptr inbounds float, float* %2, i64 %64
%i648B

	full_text
	
i64 %64
Lload8BB
@
	full_text3
1
/%66 = load float, float* %65, align 4, !tbaa !8
+float*8B

	full_text


float* %65
6fadd8B,
*
	full_text

%67 = fadd float %61, %66
)float8B

	full_text

	float %61
)float8B

	full_text

	float %66
6add8B-
+
	full_text

%68 = add nsw i32 %56, %10
%i328B

	full_text
	
i32 %56
%i328B

	full_text
	
i32 %10
6sext8B,
*
	full_text

%69 = sext i32 %68 to i64
%i328B

	full_text
	
i32 %68
\getelementptr8BI
G
	full_text:
8
6%70 = getelementptr inbounds float, float* %2, i64 %69
%i648B

	full_text
	
i64 %69
Lload8BB
@
	full_text3
1
/%71 = load float, float* %70, align 4, !tbaa !8
+float*8B

	full_text


float* %70
5fmul8B+
)
	full_text

%72 = fmul float %71, %0
)float8B

	full_text

	float %71
@fsub8B6
4
	full_text'
%
#%73 = fsub float -0.000000e+00, %72
)float8B

	full_text

	float %72
dcall8BZ
X
	full_textK
I
G%74 = tail call float @llvm.fmuladd.f32(float %1, float %67, float %73)
)float8B

	full_text

	float %67
)float8B

	full_text

	float %73
\getelementptr8BI
G
	full_text:
8
6%75 = getelementptr inbounds float, float* %3, i64 %69
%i648B

	full_text
	
i64 %69
Lstore8BA
?
	full_text2
0
.store float %74, float* %75, align 4, !tbaa !8
)float8B

	full_text

	float %74
+float*8B

	full_text


float* %75
'br8B

	full_text

br label %76
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %5
(float8B

	full_text


float %1
*float*8B

	full_text

	float* %2
*float*8B

	full_text

	float* %3
$i328B

	full_text


i32 %4
(float8B

	full_text


float %0
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
$i328B

	full_text


i32 -1
#i328B

	full_text	

i32 2
3float8B&
$
	full_text

float -0.000000e+00
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 0        	
 		                      !  "# "" $% $$ &' && () (( *+ ** ,- ,. ,, /0 // 12 13 11 45 44 67 66 89 88 :; :< :: => == ?@ ?? AB AC AA DE DD FG FH FF IJ II KL KK MN MM OP OQ OO RS RR TU TV TT WX WW YZ Y[ YY \] \\ ^_ ^^ `a `` bc bd bb ef ee gh gi gg jk jj lm ln ll op oo qr qq st ss uv uw uu xy xx z{ z| zz }~ }} 	Ä  ÅÇ ÅÅ ÉÑ É
Ö ÉÉ Üá Ü
à ÜÜ âä ââ ã
å ãã çé çç èê èè ë
í ëë ì
î ì
ï ìì ñ
ó ññ òô ò
ö òò õ	ù 	ù *	ù ?û ìü $ü 6ü Kü ^ü qü ü ã† ñ° 	° 	° /	° D	° W	° j
¢ è    
        	     ! #" %$ ' )( +* -	 ., 0/ 2 31 54 76 9& ;8 <	 > @= B? CA ED G HF JI LK N: PM Q	 SR U? VT XW Z [Y ]\ _^ aO c` d f? h	 ig ke mj nl po rq tb vs w yx {j |z ~} Ä Çu ÑÅ Öj á àÜ äâ åã éç êè íÉ îë ïâ óì ôñ ö  úõ ú ££ §§ ú ££  ££ ì §§ ì ££ 	• 	• (	• R	• x¶ ß ë	® 	® 	® © 	© 	© =	© e™ "
naive_kernel"
_Z13get_global_idj"
llvm.fmuladd.f32*ú
#parboil-0.2-stencil-naive_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

wgsize
Ä

wgsize_log1p
D∏ïA
 
transfer_bytes_log1p
D∏ïA

devmap_label


transfer_bytes
ÄÄÄ@