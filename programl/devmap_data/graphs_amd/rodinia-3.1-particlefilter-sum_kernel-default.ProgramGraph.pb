

[external]
KcallBC
A
	full_text4
2
0%3 = tail call i64 @_Z13get_global_idj(i32 0) #2
4truncB+
)
	full_text

%4 = trunc i64 %3 to i32
"i64B

	full_text


i64 %3
1icmpB)
'
	full_text

%5 = icmp eq i32 %4, 0
"i32B

	full_text


i32 %4
6brB0
.
	full_text!

br i1 %5, label %6, label %77
 i1B

	full_text	

i1 %5
Ncall8BD
B
	full_text5
3
1%7 = tail call i64 @_Z14get_local_sizej(i32 0) #2
:sitofp8B.
,
	full_text

%8 = sitofp i32 %1 to float
:uitofp8B.
,
	full_text

%9 = uitofp i64 %7 to float
$i648B

	full_text


i64 %7
@fdiv8B6
4
	full_text'
%
#%10 = fdiv float %8, %9, !fpmath !7
(float8B

	full_text


float %8
(float8B

	full_text


float %9
Jcall8B@
>
	full_text1
/
-%11 = tail call float @_Z4ceilf(float %10) #2
)float8B

	full_text

	float %10
<fptosi8B0
.
	full_text!

%12 = fptosi float %11 to i32
)float8B

	full_text

	float %11
6icmp8B,
*
	full_text

%13 = icmp sgt i32 %12, 0
%i328B

	full_text
	
i32 %12
:br8B2
0
	full_text#
!
br i1 %13, label %14, label %75
#i18B

	full_text


i1 %13
6zext8B,
*
	full_text

%15 = zext i32 %12 to i64
%i328B

	full_text
	
i32 %12
5add8B,
*
	full_text

%16 = add nsw i64 %15, -1
%i648B

	full_text
	
i64 %15
0and8B'
%
	full_text

%17 = and i64 %15, 7
%i648B

	full_text
	
i64 %15
6icmp8B,
*
	full_text

%18 = icmp ult i64 %16, 7
%i648B

	full_text
	
i64 %16
:br8B2
0
	full_text#
!
br i1 %18, label %59, label %19
#i18B

	full_text


i1 %18
6sub8B-
+
	full_text

%20 = sub nsw i64 %15, %17
%i648B

	full_text
	
i64 %15
%i648B

	full_text
	
i64 %17
'br8B

	full_text

br label %21
Bphi8B9
7
	full_text*
(
&%22 = phi i64 [ 0, %19 ], [ %56, %21 ]
%i648B

	full_text
	
i64 %56
Ophi8BF
D
	full_text7
5
3%23 = phi float [ 0.000000e+00, %19 ], [ %55, %21 ]
)float8B

	full_text

	float %55
Dphi8B;
9
	full_text,
*
(%24 = phi i64 [ %20, %19 ], [ %57, %21 ]
%i648B

	full_text
	
i64 %20
%i648B

	full_text
	
i64 %57
\getelementptr8BI
G
	full_text:
8
6%25 = getelementptr inbounds float, float* %0, i64 %22
%i648B

	full_text
	
i64 %22
Lload8BB
@
	full_text3
1
/%26 = load float, float* %25, align 4, !tbaa !8
+float*8B

	full_text


float* %25
6fadd8B,
*
	full_text

%27 = fadd float %23, %26
)float8B

	full_text

	float %23
)float8B

	full_text

	float %26
.or8B&
$
	full_text

%28 = or i64 %22, 1
%i648B

	full_text
	
i64 %22
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
/%30 = load float, float* %29, align 4, !tbaa !8
+float*8B

	full_text


float* %29
6fadd8B,
*
	full_text

%31 = fadd float %27, %30
)float8B

	full_text

	float %27
)float8B

	full_text

	float %30
.or8B&
$
	full_text

%32 = or i64 %22, 2
%i648B

	full_text
	
i64 %22
\getelementptr8BI
G
	full_text:
8
6%33 = getelementptr inbounds float, float* %0, i64 %32
%i648B

	full_text
	
i64 %32
Lload8BB
@
	full_text3
1
/%34 = load float, float* %33, align 4, !tbaa !8
+float*8B

	full_text


float* %33
6fadd8B,
*
	full_text

%35 = fadd float %31, %34
)float8B

	full_text

	float %31
)float8B

	full_text

	float %34
.or8B&
$
	full_text

%36 = or i64 %22, 3
%i648B

	full_text
	
i64 %22
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
/%38 = load float, float* %37, align 4, !tbaa !8
+float*8B

	full_text


float* %37
6fadd8B,
*
	full_text

%39 = fadd float %35, %38
)float8B

	full_text

	float %35
)float8B

	full_text

	float %38
.or8B&
$
	full_text

%40 = or i64 %22, 4
%i648B

	full_text
	
i64 %22
\getelementptr8BI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %0, i64 %40
%i648B

	full_text
	
i64 %40
Lload8BB
@
	full_text3
1
/%42 = load float, float* %41, align 4, !tbaa !8
+float*8B

	full_text


float* %41
6fadd8B,
*
	full_text

%43 = fadd float %39, %42
)float8B

	full_text

	float %39
)float8B

	full_text

	float %42
.or8B&
$
	full_text

%44 = or i64 %22, 5
%i648B

	full_text
	
i64 %22
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
/%46 = load float, float* %45, align 4, !tbaa !8
+float*8B

	full_text


float* %45
6fadd8B,
*
	full_text

%47 = fadd float %43, %46
)float8B

	full_text

	float %43
)float8B

	full_text

	float %46
.or8B&
$
	full_text

%48 = or i64 %22, 6
%i648B

	full_text
	
i64 %22
\getelementptr8BI
G
	full_text:
8
6%49 = getelementptr inbounds float, float* %0, i64 %48
%i648B

	full_text
	
i64 %48
Lload8BB
@
	full_text3
1
/%50 = load float, float* %49, align 4, !tbaa !8
+float*8B

	full_text


float* %49
6fadd8B,
*
	full_text

%51 = fadd float %47, %50
)float8B

	full_text

	float %47
)float8B

	full_text

	float %50
.or8B&
$
	full_text

%52 = or i64 %22, 7
%i648B

	full_text
	
i64 %22
\getelementptr8BI
G
	full_text:
8
6%53 = getelementptr inbounds float, float* %0, i64 %52
%i648B

	full_text
	
i64 %52
Lload8BB
@
	full_text3
1
/%54 = load float, float* %53, align 4, !tbaa !8
+float*8B

	full_text


float* %53
6fadd8B,
*
	full_text

%55 = fadd float %51, %54
)float8B

	full_text

	float %51
)float8B

	full_text

	float %54
4add8B+
)
	full_text

%56 = add nsw i64 %22, 8
%i648B

	full_text
	
i64 %22
1add8B(
&
	full_text

%57 = add i64 %24, -8
%i648B

	full_text
	
i64 %24
5icmp8B+
)
	full_text

%58 = icmp eq i64 %57, 0
%i648B

	full_text
	
i64 %57
:br8B2
0
	full_text#
!
br i1 %58, label %59, label %21
#i18B

	full_text


i1 %58
Hphi8B?
=
	full_text0
.
,%60 = phi float [ undef, %14 ], [ %55, %21 ]
)float8B

	full_text

	float %55
Bphi8B9
7
	full_text*
(
&%61 = phi i64 [ 0, %14 ], [ %56, %21 ]
%i648B

	full_text
	
i64 %56
Ophi8BF
D
	full_text7
5
3%62 = phi float [ 0.000000e+00, %14 ], [ %55, %21 ]
)float8B

	full_text

	float %55
5icmp8B+
)
	full_text

%63 = icmp eq i64 %17, 0
%i648B

	full_text
	
i64 %17
:br8B2
0
	full_text#
!
br i1 %63, label %75, label %64
#i18B

	full_text


i1 %63
'br8B

	full_text

br label %65
Dphi8B;
9
	full_text,
*
(%66 = phi i64 [ %61, %64 ], [ %72, %65 ]
%i648B

	full_text
	
i64 %61
%i648B

	full_text
	
i64 %72
Fphi8B=
;
	full_text.
,
*%67 = phi float [ %62, %64 ], [ %71, %65 ]
)float8B

	full_text

	float %62
)float8B

	full_text

	float %71
Dphi8B;
9
	full_text,
*
(%68 = phi i64 [ %17, %64 ], [ %73, %65 ]
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %73
\getelementptr8BI
G
	full_text:
8
6%69 = getelementptr inbounds float, float* %0, i64 %66
%i648B

	full_text
	
i64 %66
Lload8BB
@
	full_text3
1
/%70 = load float, float* %69, align 4, !tbaa !8
+float*8B

	full_text


float* %69
6fadd8B,
*
	full_text

%71 = fadd float %67, %70
)float8B

	full_text

	float %67
)float8B

	full_text

	float %70
8add8B/
-
	full_text 

%72 = add nuw nsw i64 %66, 1
%i648B

	full_text
	
i64 %66
1add8B(
&
	full_text

%73 = add i64 %68, -1
%i648B

	full_text
	
i64 %68
5icmp8B+
)
	full_text

%74 = icmp eq i64 %73, 0
%i648B

	full_text
	
i64 %73
Jbr8BB
@
	full_text3
1
/br i1 %74, label %75, label %65, !llvm.loop !12
#i18B

	full_text


i1 %74
\phi8BS
Q
	full_textD
B
@%76 = phi float [ 0.000000e+00, %6 ], [ %60, %59 ], [ %71, %65 ]
)float8B

	full_text

	float %60
)float8B

	full_text

	float %71
Kstore8B@
>
	full_text1
/
-store float %76, float* %0, align 4, !tbaa !8
)float8B

	full_text

	float %76
'br8B

	full_text

br label %77
$ret8	B

	full_text


ret void
$i328
B

	full_text


i32 %1
*float*8
B
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
-; undefined function B

	full_text

 
#i648
B

	full_text	

i64 0
2float8
B%
#
	full_text

float 0.000000e+00
#i648
B

	full_text	

i64 1
+float8
B

	full_text

float undef
$i648
B

	full_text


i64 -8
#i648
B

	full_text	

i64 4
#i328
B

	full_text	

i32 0
#i648
B

	full_text	

i64 2
#i648
B

	full_text	

i64 5
#i648
B

	full_text	

i64 6
#i648
B

	full_text	

i64 3
#i648
B

	full_text	

i64 8
$i648
B

	full_text


i64 -1
#i648
B

	full_text	

i64 7       		 
 

                     " !# !! $& %% '( '' )* )+ )) ,- ,, ./ .. 01 02 00 34 33 56 55 78 77 9: 9; 99 <= << >? >> @A @@ BC BD BB EF EE GH GG IJ II KL KM KK NO NN PQ PP RS RR TU TV TT WX WW YZ YY [\ [[ ]^ ]_ ]] `a `` bc bb de dd fg fh ff ij ii kl kk mn mm op oq oo rs rr tu tt vw vv xy x{ zz |} || ~ ~~ ÄÅ ÄÄ ÇÉ ÇÜ Ö
á ÖÖ àâ à
ä àà ãå ã
ç ãã é
è éé êë êê íì í
î íí ïñ ïï óò óó ôö ôô õú õ
û ù
ü ùù †° †† ¢§ 	• ,• 5• >• G• P• Y• b• k• é
• †    	 
            " #r &o (! *t +% -, /' 1. 2% 43 65 80 :7 ;% =< ?> A9 C@ D% FE HG JB LI M% ON QP SK UR V% XW ZY \T ^[ _% a` cb e] gd h% ji lk nf pm q% s) ut wv yo {r }o  ÅÄ É| Üï á~ âí ä åó çÖ èé ëà ìê îÖ ñã òó öô úz ûí üù °  £  ù z !¢ £Ç ùÇ Ñ$ %Ñ Öx zx %õ ùõ Ö ¶¶ ®® ßß £ ®®  ßß  ¶¶ © %	© v© |
© Ä
© ô™ '™ ~™ ù	´ 3
´ ï¨ z	≠ t	Æ NØ 	Ø Ø 	Ø 	∞ <	± W	≤ `	≥ E	¥ r	µ 
µ ó	∂ 	∂ 	∂ i"

sum_kernel"
_Z13get_global_idj"
_Z14get_local_sizej"

_Z4ceilf*°
(rodinia-3.1-particlefilter-sum_kernel.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å
 
transfer_bytes_log1p
@ïA

devmap_label
 

wgsize
Ä

transfer_bytes
©¨<

wgsize_log1p
@ïA