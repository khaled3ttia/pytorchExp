

[external]
6allocaB,
*
	full_text

%11 = alloca i64, align 8
AbitcastB6
4
	full_text'
%
#%12 = bitcast i64* %11 to [8 x i8]*
%i64*B

	full_text


i64* %11
=allocaB3
1
	full_text$
"
 %13 = alloca [4 x i32], align 16
KcallBC
A
	full_text4
2
0%14 = tail call i64 @_Z12get_group_idj(i32 0) #4
McallBE
C
	full_text6
4
2%15 = tail call i64 @_Z14get_local_sizej(i32 0) #4
0mulB)
'
	full_text

%16 = mul i64 %15, %14
#i64B

	full_text
	
i64 %15
#i64B

	full_text
	
i64 %14
KcallBC
A
	full_text4
2
0%17 = tail call i64 @_Z12get_local_idj(i32 0) #4
0addB)
'
	full_text

%18 = add i64 %16, %17
#i64B

	full_text
	
i64 %16
#i64B

	full_text
	
i64 %17
6truncB-
+
	full_text

%19 = trunc i64 %18 to i32
#i64B

	full_text
	
i64 %18
3mulB,
*
	full_text

%20 = mul nsw i32 %19, %6
#i32B

	full_text
	
i32 %19
;bitcastB0
.
	full_text!

%21 = bitcast i64* %11 to i8*
%i64*B

	full_text


i64* %11
YcallBQ
O
	full_textB
@
>call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %21) #5
#i8*B

	full_text
	
i8* %21
1uremB)
'
	full_text

%22 = urem i32 %20, %6
#i32B

	full_text
	
i32 %20
5truncB,
*
	full_text

%23 = trunc i32 %22 to i8
#i32B

	full_text
	
i32 %22
DstoreB;
9
	full_text,
*
(store i8 %23, i8* %21, align 8, !tbaa !8
!i8B

	full_text


i8 %23
#i8*B

	full_text
	
i8* %21
1udivB)
'
	full_text

%24 = udiv i32 %20, %6
#i32B

	full_text
	
i32 %20
1uremB)
'
	full_text

%25 = urem i32 %24, %6
#i32B

	full_text
	
i32 %24
5truncB,
*
	full_text

%26 = trunc i32 %25 to i8
#i32B

	full_text
	
i32 %25
SgetelementptrBB
@
	full_text3
1
/%27 = getelementptr inbounds i8, i8* %21, i64 1
#i8*B

	full_text
	
i8* %21
DstoreB;
9
	full_text,
*
(store i8 %26, i8* %27, align 1, !tbaa !8
!i8B

	full_text


i8 %26
#i8*B

	full_text
	
i8* %27
1udivB)
'
	full_text

%28 = udiv i32 %24, %6
#i32B

	full_text
	
i32 %24
1uremB)
'
	full_text

%29 = urem i32 %28, %6
#i32B

	full_text
	
i32 %28
5truncB,
*
	full_text

%30 = trunc i32 %29 to i8
#i32B

	full_text
	
i32 %29
SgetelementptrBB
@
	full_text3
1
/%31 = getelementptr inbounds i8, i8* %21, i64 2
#i8*B

	full_text
	
i8* %21
DstoreB;
9
	full_text,
*
(store i8 %30, i8* %31, align 2, !tbaa !8
!i8B

	full_text


i8 %30
#i8*B

	full_text
	
i8* %31
1udivB)
'
	full_text

%32 = udiv i32 %28, %6
#i32B

	full_text
	
i32 %28
1uremB)
'
	full_text

%33 = urem i32 %32, %6
#i32B

	full_text
	
i32 %32
5truncB,
*
	full_text

%34 = trunc i32 %33 to i8
#i32B

	full_text
	
i32 %33
SgetelementptrBB
@
	full_text3
1
/%35 = getelementptr inbounds i8, i8* %21, i64 3
#i8*B

	full_text
	
i8* %21
DstoreB;
9
	full_text,
*
(store i8 %34, i8* %35, align 1, !tbaa !8
!i8B

	full_text


i8 %34
#i8*B

	full_text
	
i8* %35
1udivB)
'
	full_text

%36 = udiv i32 %32, %6
#i32B

	full_text
	
i32 %32
1uremB)
'
	full_text

%37 = urem i32 %36, %6
#i32B

	full_text
	
i32 %36
5truncB,
*
	full_text

%38 = trunc i32 %37 to i8
#i32B

	full_text
	
i32 %37
SgetelementptrBB
@
	full_text3
1
/%39 = getelementptr inbounds i8, i8* %21, i64 4
#i8*B

	full_text
	
i8* %21
DstoreB;
9
	full_text,
*
(store i8 %38, i8* %39, align 4, !tbaa !8
!i8B

	full_text


i8 %38
#i8*B

	full_text
	
i8* %39
1udivB)
'
	full_text

%40 = udiv i32 %36, %6
#i32B

	full_text
	
i32 %36
1uremB)
'
	full_text

%41 = urem i32 %40, %6
#i32B

	full_text
	
i32 %40
5truncB,
*
	full_text

%42 = trunc i32 %41 to i8
#i32B

	full_text
	
i32 %41
SgetelementptrBB
@
	full_text3
1
/%43 = getelementptr inbounds i8, i8* %21, i64 5
#i8*B

	full_text
	
i8* %21
DstoreB;
9
	full_text,
*
(store i8 %42, i8* %43, align 1, !tbaa !8
!i8B

	full_text


i8 %42
#i8*B

	full_text
	
i8* %43
1udivB)
'
	full_text

%44 = udiv i32 %40, %6
#i32B

	full_text
	
i32 %40
1uremB)
'
	full_text

%45 = urem i32 %44, %6
#i32B

	full_text
	
i32 %44
5truncB,
*
	full_text

%46 = trunc i32 %45 to i8
#i32B

	full_text
	
i32 %45
SgetelementptrBB
@
	full_text3
1
/%47 = getelementptr inbounds i8, i8* %21, i64 6
#i8*B

	full_text
	
i8* %21
DstoreB;
9
	full_text,
*
(store i8 %46, i8* %47, align 2, !tbaa !8
!i8B

	full_text


i8 %46
#i8*B

	full_text
	
i8* %47
1udivB)
'
	full_text

%48 = udiv i32 %44, %6
#i32B

	full_text
	
i32 %44
1uremB)
'
	full_text

%49 = urem i32 %48, %6
#i32B

	full_text
	
i32 %48
5truncB,
*
	full_text

%50 = trunc i32 %49 to i8
#i32B

	full_text
	
i32 %49
SgetelementptrBB
@
	full_text3
1
/%51 = getelementptr inbounds i8, i8* %21, i64 7
#i8*B

	full_text
	
i8* %21
DstoreB;
9
	full_text,
*
(store i8 %50, i8* %51, align 1, !tbaa !8
!i8B

	full_text


i8 %50
#i8*B

	full_text
	
i8* %51
3icmpB+
)
	full_text

%52 = icmp sgt i32 %6, 0
8brB2
0
	full_text#
!
br i1 %52, label %53, label %81
!i1B

	full_text


i1 %52
Cbitcast8B6
4
	full_text'
%
#%54 = bitcast [4 x i32]* %13 to i8*
3
[4 x i32]*8B!

	full_text

[4 x i32]* %13
>bitcast8B1
/
	full_text"
 
%55 = bitcast i64* %11 to i32*
'i64*8B

	full_text


i64* %11
jgetelementptr8BW
U
	full_textH
F
D%56 = getelementptr inbounds [4 x i32], [4 x i32]* %13, i64 0, i64 0
3
[4 x i32]*8B!

	full_text

[4 x i32]* %13
jgetelementptr8BW
U
	full_textH
F
D%57 = getelementptr inbounds [4 x i32], [4 x i32]* %13, i64 0, i64 1
3
[4 x i32]*8B!

	full_text

[4 x i32]* %13
jgetelementptr8BW
U
	full_textH
F
D%58 = getelementptr inbounds [4 x i32], [4 x i32]* %13, i64 0, i64 2
3
[4 x i32]*8B!

	full_text

[4 x i32]* %13
jgetelementptr8BW
U
	full_textH
F
D%59 = getelementptr inbounds [4 x i32], [4 x i32]* %13, i64 0, i64 3
3
[4 x i32]*8B!

	full_text

[4 x i32]* %13
hgetelementptr8BU
S
	full_textF
D
B%60 = getelementptr inbounds [8 x i8], [8 x i8]* %12, i64 0, i64 1
1	[8 x i8]*8B 

	full_text

[8 x i8]* %12
Tgetelementptr8BA
?
	full_text2
0
.%61 = getelementptr inbounds i8, i8* %8, i64 1
hgetelementptr8BU
S
	full_textF
D
B%62 = getelementptr inbounds [8 x i8], [8 x i8]* %12, i64 0, i64 2
1	[8 x i8]*8B 

	full_text

[8 x i8]* %12
Tgetelementptr8BA
?
	full_text2
0
.%63 = getelementptr inbounds i8, i8* %8, i64 2
hgetelementptr8BU
S
	full_textF
D
B%64 = getelementptr inbounds [8 x i8], [8 x i8]* %12, i64 0, i64 3
1	[8 x i8]*8B 

	full_text

[8 x i8]* %12
Tgetelementptr8BA
?
	full_text2
0
.%65 = getelementptr inbounds i8, i8* %8, i64 3
hgetelementptr8BU
S
	full_textF
D
B%66 = getelementptr inbounds [8 x i8], [8 x i8]* %12, i64 0, i64 4
1	[8 x i8]*8B 

	full_text

[8 x i8]* %12
Tgetelementptr8BA
?
	full_text2
0
.%67 = getelementptr inbounds i8, i8* %8, i64 4
hgetelementptr8BU
S
	full_textF
D
B%68 = getelementptr inbounds [8 x i8], [8 x i8]* %12, i64 0, i64 5
1	[8 x i8]*8B 

	full_text

[8 x i8]* %12
Tgetelementptr8BA
?
	full_text2
0
.%69 = getelementptr inbounds i8, i8* %8, i64 5
hgetelementptr8BU
S
	full_textF
D
B%70 = getelementptr inbounds [8 x i8], [8 x i8]* %12, i64 0, i64 6
1	[8 x i8]*8B 

	full_text

[8 x i8]* %12
Tgetelementptr8BA
?
	full_text2
0
.%71 = getelementptr inbounds i8, i8* %8, i64 6
hgetelementptr8BU
S
	full_textF
D
B%72 = getelementptr inbounds [8 x i8], [8 x i8]* %12, i64 0, i64 7
1	[8 x i8]*8B 

	full_text

[8 x i8]* %12
Tgetelementptr8BA
?
	full_text2
0
.%73 = getelementptr inbounds i8, i8* %8, i64 7
Vgetelementptr8BC
A
	full_text4
2
0%74 = getelementptr inbounds i32, i32* %9, i64 1
Vgetelementptr8BC
A
	full_text4
2
0%75 = getelementptr inbounds i32, i32* %9, i64 2
Vgetelementptr8BC
A
	full_text4
2
0%76 = getelementptr inbounds i32, i32* %9, i64 3
'br8B

	full_text

br label %77
Dphi8B;
9
	full_text,
*
(%78 = phi i32 [ 0, %53 ], [ %106, %103 ]
&i328B

	full_text


i32 %106
6add8B-
+
	full_text

%79 = add nsw i32 %78, %20
%i328B

	full_text
	
i32 %78
%i328B

	full_text
	
i32 %20
7icmp8B-
+
	full_text

%80 = icmp slt i32 %79, %4
%i328B

	full_text
	
i32 %79
:br8B2
0
	full_text#
!
br i1 %80, label %82, label %81
#i18B

	full_text


i1 %80
Ycall8BO
M
	full_text@
>
<call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %21) #5
%i8*8B

	full_text
	
i8* %21
$ret8B

	full_text


ret void
\call8BR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %54) #5
%i8*8B

	full_text
	
i8* %54
acall8BW
U
	full_textH
F
Dcall void @md5_2words(i32* nonnull %55, i32 %5, i32* nonnull %56) #5
'i32*8B

	full_text


i32* %55
'i32*8B

	full_text


i32* %56
Jload8B@
>
	full_text1
/
-%83 = load i32, i32* %56, align 16, !tbaa !11
'i32*8B

	full_text


i32* %56
6icmp8B,
*
	full_text

%84 = icmp eq i32 %83, %0
%i328B

	full_text
	
i32 %83
>load8B4
2
	full_text%
#
!%85 = load i32, i32* %57, align 4
'i32*8B

	full_text


i32* %57
6icmp8B,
*
	full_text

%86 = icmp eq i32 %85, %1
%i328B

	full_text
	
i32 %85
1and8B(
&
	full_text

%87 = and i1 %84, %86
#i18B

	full_text


i1 %84
#i18B

	full_text


i1 %86
>load8B4
2
	full_text%
#
!%88 = load i32, i32* %58, align 8
'i32*8B

	full_text


i32* %58
6icmp8B,
*
	full_text

%89 = icmp eq i32 %88, %2
%i328B

	full_text
	
i32 %88
1and8B(
&
	full_text

%90 = and i1 %87, %89
#i18B

	full_text


i1 %87
#i18B

	full_text


i1 %89
>load8B4
2
	full_text%
#
!%91 = load i32, i32* %59, align 4
'i32*8B

	full_text


i32* %59
6icmp8B,
*
	full_text

%92 = icmp eq i32 %91, %3
%i328B

	full_text
	
i32 %91
1and8B(
&
	full_text

%93 = and i1 %90, %92
#i18B

	full_text


i1 %90
#i18B

	full_text


i1 %92
;br8B3
1
	full_text$
"
 br i1 %93, label %94, label %103
#i18B

	full_text


i1 %93
Hstore8B=
;
	full_text.
,
*store i32 %79, i32* %7, align 4, !tbaa !11
%i328B

	full_text
	
i32 %79
Fload8B<
:
	full_text-
+
)%95 = load i8, i8* %21, align 8, !tbaa !8
%i8*8B

	full_text
	
i8* %21
Estore8B:
8
	full_text+
)
'store i8 %95, i8* %8, align 1, !tbaa !8
#i88B

	full_text


i8 %95
Fload8B<
:
	full_text-
+
)%96 = load i8, i8* %60, align 1, !tbaa !8
%i8*8B

	full_text
	
i8* %60
Fstore8B;
9
	full_text,
*
(store i8 %96, i8* %61, align 1, !tbaa !8
#i88B

	full_text


i8 %96
%i8*8B

	full_text
	
i8* %61
Fload8B<
:
	full_text-
+
)%97 = load i8, i8* %62, align 2, !tbaa !8
%i8*8B

	full_text
	
i8* %62
Fstore8B;
9
	full_text,
*
(store i8 %97, i8* %63, align 1, !tbaa !8
#i88B

	full_text


i8 %97
%i8*8B

	full_text
	
i8* %63
Fload8B<
:
	full_text-
+
)%98 = load i8, i8* %64, align 1, !tbaa !8
%i8*8B

	full_text
	
i8* %64
Fstore8B;
9
	full_text,
*
(store i8 %98, i8* %65, align 1, !tbaa !8
#i88B

	full_text


i8 %98
%i8*8B

	full_text
	
i8* %65
Fload8B<
:
	full_text-
+
)%99 = load i8, i8* %66, align 4, !tbaa !8
%i8*8B

	full_text
	
i8* %66
Fstore8B;
9
	full_text,
*
(store i8 %99, i8* %67, align 1, !tbaa !8
#i88B

	full_text


i8 %99
%i8*8B

	full_text
	
i8* %67
Gload8B=
;
	full_text.
,
*%100 = load i8, i8* %68, align 1, !tbaa !8
%i8*8B

	full_text
	
i8* %68
Gstore8B<
:
	full_text-
+
)store i8 %100, i8* %69, align 1, !tbaa !8
$i88B

	full_text
	
i8 %100
%i8*8B

	full_text
	
i8* %69
Gload8B=
;
	full_text.
,
*%101 = load i8, i8* %70, align 2, !tbaa !8
%i8*8B

	full_text
	
i8* %70
Gstore8B<
:
	full_text-
+
)store i8 %101, i8* %71, align 1, !tbaa !8
$i88B

	full_text
	
i8 %101
%i8*8B

	full_text
	
i8* %71
Gload8B=
;
	full_text.
,
*%102 = load i8, i8* %72, align 1, !tbaa !8
%i8*8B

	full_text
	
i8* %72
Gstore8B<
:
	full_text-
+
)store i8 %102, i8* %73, align 1, !tbaa !8
$i88B

	full_text
	
i8 %102
%i8*8B

	full_text
	
i8* %73
Gstore8B<
:
	full_text-
+
)store i32 %0, i32* %9, align 4, !tbaa !11
Hstore8B=
;
	full_text.
,
*store i32 %1, i32* %74, align 4, !tbaa !11
'i32*8B

	full_text


i32* %74
Hstore8B=
;
	full_text.
,
*store i32 %2, i32* %75, align 4, !tbaa !11
'i32*8B

	full_text


i32* %75
Hstore8B=
;
	full_text.
,
*store i32 %3, i32* %76, align 4, !tbaa !11
'i32*8B

	full_text


i32* %76
(br8B 

	full_text

br label %103
Gload8B=
;
	full_text.
,
*%104 = load i8, i8* %21, align 8, !tbaa !8
%i8*8B

	full_text
	
i8* %21
1add8B(
&
	full_text

%105 = add i8 %104, 1
$i88B

	full_text
	
i8 %104
Gstore8B<
:
	full_text-
+
)store i8 %105, i8* %21, align 8, !tbaa !8
$i88B

	full_text
	
i8 %105
%i8*8B

	full_text
	
i8* %21
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %54) #5
%i8*8B

	full_text
	
i8* %54
9add8B0
.
	full_text!

%106 = add nuw nsw i32 %78, 1
%i328B

	full_text
	
i32 %78
9icmp8B/
-
	full_text 

%107 = icmp slt i32 %106, %6
&i328B

	full_text


i32 %106
;br8B3
1
	full_text$
"
 br i1 %107, label %77, label %81
$i18B

	full_text
	
i1 %107
$i328B

	full_text


i32 %6
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %1
$i328B

	full_text


i32 %0
$i8*8B

	full_text


i8* %8
&i32*8B

	full_text
	
i32* %7
$i328B

	full_text


i32 %4
&i32*8B

	full_text
	
i32* %9
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
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
!i88B

	full_text

i8 1
#i648B

	full_text	

i64 6
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 5
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 4
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 7
$i648B

	full_text


i64 16
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 2
#i648B

	full_text	

i64 8       	  

                        !" !! #$ ## %& %' %% () (( *+ ** ,- ,, ./ .. 01 02 00 34 33 56 55 78 77 9: 99 ;< ;= ;; >? >> @A @@ BC BB DE DD FG FH FF IJ II KL KK MN MM OP OO QR QS QQ TU TT VW VV XY XX Z[ ZZ \] \^ \\ _` __ ab aa cd cc ef ee gh gi gg jj kl kn mm op oo qr qq st ss uv uu wx ww yz yy {{ |} || ~~ Ä  ÅÅ ÇÉ ÇÇ ÑÑ ÖÜ ÖÖ áá àâ àà ää ãå ãã çç éé èè êê ë
ì íí îï î
ñ îî óò óó ôö ô
ú õõ ù
ü ûû †° †
¢ †† £§ ££ •¶ •• ß® ßß ©™ ©© ´¨ ´
≠ ´´ ÆØ ÆÆ ∞± ∞∞ ≤≥ ≤
¥ ≤≤ µ∂ µµ ∑∏ ∑∑ π∫ π
ª ππ ºΩ ºø ææ ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒƒ ∆« ∆
» ∆∆ …  …… ÀÃ À
Õ ÀÀ Œœ ŒŒ –— –
“ –– ”‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›› ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ ÁÁ Ë
È ËË Í
Î ÍÍ Ï
Ì ÏÏ Ó ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ
˜ ˆˆ ¯˘ ¯¯ ˙˚ ˙˙ ¸˝ ¸	˛ 	˛ 	˛ 	˛ 	˛ (	˛ *	˛ 3	˛ 5	˛ >	˛ @	˛ I	˛ K	˛ T	˛ V	˛ _	˛ a˛ j
˛ ˙
ˇ ∞ˇ Í
Ä ∑Ä Ï
Å †
Ç ©Ç Ë
É •É ÁÑ {Ñ ~Ñ ÅÑ ÑÑ áÑ äÑ ç
Ñ ¬
Ö æ
Ü óá éá èá ê
á Á   	 
             " $! &# ' )( +* - /, 1. 2( 43 65 8 :7 <9 =3 ?> A@ C EB GD H> JI LK N PM RO SI UT WV Y [X ]Z ^T `_ ba d fc he ij l n p r t v x z } Ä É Ü â å¯ ìí ï ñî òó ö úm üo °q ¢q §£ ¶s ®ß ™• ¨© ≠u ØÆ ±´ ≥∞ ¥w ∂µ ∏≤ ∫∑ ªπ Ωî ø ¡¿ √y ≈ƒ «{ »|  … Ã~ Õ œŒ —Å “Ç ‘” ÷Ñ ◊Ö Ÿÿ €á ‹à ﬁ› ‡ä ·ã „‚ Âç Êé Èè Îê Ì Ô ÚÒ Ù ım ˜í ˘¯ ˚˙ ˝k mk õë íô ûô õº æº ÔÓ Ô¸ í¸ õ ù àà ââ ää ãã åå çç
 ãã 
ˆ çç ˆõ çç õ ââ û àà û† åå † ää  àà 
é Ò	è Z
è à
è äê ê ê 
	ê jê í	ë 9	ë w	ë 
ë Å
ë ê	í O
í Ö
í á	ì #	ì s	ì y	ì {
ì é	î D
î Ç
î Ñï ï 
ï ¯	ñ e
ñ ã
ñ çó ûó ˆ	ò q	ò q	ò s	ò u	ò w	ò y	ò |	ò 
ò Ç
ò Ö
ò à
ò ã	ô .	ô u	ô |	ô ~
ô èö ö õ"
FindKeyWithDigest_Kernel"
llvm.lifetime.start.p0i8"
_Z12get_group_idj"
_Z14get_local_sizej"
_Z12get_local_idj"

md5_2words"
llvm.lifetime.end.p0i8*ß
.shoc-1.1.5-MD5Hash-FindKeyWithDigest_Kernel.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282

wgsize_log1p
gm¥@

wgsize
Ä

devmap_label

 
transfer_bytes_log1p
gm¥@

transfer_bytes
ò