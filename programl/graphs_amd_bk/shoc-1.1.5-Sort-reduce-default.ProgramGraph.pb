

[external]
=allocaB3
1
	full_text$
"
 %6 = alloca [16 x i32], align 16
.sdivB&
$
	full_text

%7 = sdiv i32 %2, 4
2sextB*
(
	full_text

%8 = sext i32 %7 to i64
"i32B

	full_text


i32 %7
LcallBD
B
	full_text5
3
1%9 = tail call i64 @_Z14get_num_groupsj(i32 0) #4
0udivB(
&
	full_text

%10 = udiv i64 %8, %9
"i64B

	full_text


i64 %8
"i64B

	full_text


i64 %9
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
.shlB'
%
	full_text

%12 = shl i32 %11, 2
#i32B

	full_text
	
i32 %11
KcallBC
A
	full_text4
2
0%13 = tail call i64 @_Z12get_group_idj(i32 0) #4
4sextB,
*
	full_text

%14 = sext i32 %12 to i64
#i32B

	full_text
	
i32 %12
0mulB)
'
	full_text

%15 = mul i64 %13, %14
#i64B

	full_text
	
i64 %13
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
.addB'
%
	full_text

%17 = add i64 %9, -1
"i64B

	full_text


i64 %9
5icmpB-
+
	full_text

%18 = icmp eq i64 %13, %17
#i64B

	full_text
	
i64 %13
#i64B

	full_text
	
i64 %17
4addB-
+
	full_text

%19 = add nsw i32 %12, %16
#i32B

	full_text
	
i32 %12
#i32B

	full_text
	
i32 %16
AselectB7
5
	full_text(
&
$%20 = select i1 %18, i32 %2, i32 %19
!i1B

	full_text


i1 %18
#i32B

	full_text
	
i32 %19
KcallBC
A
	full_text4
2
0%21 = tail call i64 @_Z12get_local_idj(i32 0) #4
6truncB-
+
	full_text

%22 = trunc i64 %21 to i32
#i64B

	full_text
	
i64 %21
4addB-
+
	full_text

%23 = add nsw i32 %16, %22
#i32B

	full_text
	
i32 %16
#i32B

	full_text
	
i32 %22
AbitcastB6
4
	full_text'
%
#%24 = bitcast [16 x i32]* %6 to i8*
2[16 x i32]*B!

	full_text

[16 x i32]* %6
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %24) #5
#i8*B

	full_text
	
i8* %24
dcallB\
Z
	full_textM
K
Icall void @llvm.memset.p0i8.i64(i8* align 16 %24, i8 0, i64 64, i1 false)
#i8*B

	full_text
	
i8* %24
6icmpB.
,
	full_text

%25 = icmp slt i32 %23, %20
#i32B

	full_text
	
i32 %23
#i32B

	full_text
	
i32 %20
8brB2
0
	full_text#
!
br i1 %25, label %26, label %43
!i1B

	full_text


i1 %25
0and8B'
%
	full_text

%27 = and i32 %4, 31
Ocall8BE
C
	full_text6
4
2%28 = tail call i64 @_Z14get_local_sizej(i32 0) #4
'br8B

	full_text

br label %29
Dphi8B;
9
	full_text,
*
(%30 = phi i32 [ %23, %26 ], [ %41, %29 ]
%i328B

	full_text
	
i32 %23
%i328B

	full_text
	
i32 %41
6sext8B,
*
	full_text

%31 = sext i32 %30 to i64
%i328B

	full_text
	
i32 %30
Xgetelementptr8BE
C
	full_text6
4
2%32 = getelementptr inbounds i32, i32* %0, i64 %31
%i648B

	full_text
	
i64 %31
Hload8B>
<
	full_text/
-
+%33 = load i32, i32* %32, align 4, !tbaa !8
'i32*8B

	full_text


i32* %32
4lshr8B*
(
	full_text

%34 = lshr i32 %33, %27
%i328B

	full_text
	
i32 %33
%i328B

	full_text
	
i32 %27
1and8B(
&
	full_text

%35 = and i32 %34, 15
%i328B

	full_text
	
i32 %34
6zext8B,
*
	full_text

%36 = zext i32 %35 to i64
%i328B

	full_text
	
i32 %35
mgetelementptr8BZ
X
	full_textK
I
G%37 = getelementptr inbounds [16 x i32], [16 x i32]* %6, i64 0, i64 %36
4[16 x i32]*8B!

	full_text

[16 x i32]* %6
%i648B

	full_text
	
i64 %36
Hload8B>
<
	full_text/
-
+%38 = load i32, i32* %37, align 4, !tbaa !8
'i32*8B

	full_text


i32* %37
4add8B+
)
	full_text

%39 = add nsw i32 %38, 1
%i328B

	full_text
	
i32 %38
Hstore8B=
;
	full_text.
,
*store i32 %39, i32* %37, align 4, !tbaa !8
%i328B

	full_text
	
i32 %39
'i32*8B

	full_text


i32* %37
2add8B)
'
	full_text

%40 = add i64 %28, %31
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %31
8trunc8B-
+
	full_text

%41 = trunc i64 %40 to i32
%i648B

	full_text
	
i64 %40
8icmp8B.
,
	full_text

%42 = icmp sgt i32 %20, %41
%i328B

	full_text
	
i32 %20
%i328B

	full_text
	
i32 %41
:br8B2
0
	full_text#
!
br i1 %42, label %29, label %43
#i18B

	full_text


i1 %42
1shl8B(
&
	full_text

%44 = shl i64 %21, 32
%i648B

	full_text
	
i64 %21
9ashr8B/
-
	full_text 

%45 = ashr exact i64 %44, 32
%i648B

	full_text
	
i64 %44
Xgetelementptr8BE
C
	full_text6
4
2%46 = getelementptr inbounds i32, i32* %3, i64 %45
%i648B

	full_text
	
i64 %45
5icmp8B+
)
	full_text

%47 = icmp eq i32 %22, 0
%i328B

	full_text
	
i32 %22
'br8B

	full_text

br label %49
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %24) #5
%i8*8B

	full_text
	
i8* %24
$ret8B

	full_text


ret void
Bphi8B9
7
	full_text*
(
&%50 = phi i64 [ 0, %43 ], [ %78, %77 ]
%i648B

	full_text
	
i64 %78
mgetelementptr8BZ
X
	full_textK
I
G%51 = getelementptr inbounds [16 x i32], [16 x i32]* %6, i64 0, i64 %50
4[16 x i32]*8B!

	full_text

[16 x i32]* %6
%i648B

	full_text
	
i64 %50
Hload8B>
<
	full_text/
-
+%52 = load i32, i32* %51, align 4, !tbaa !8
'i32*8B

	full_text


i32* %51
Hstore8B=
;
	full_text.
,
*store i32 %52, i32* %46, align 4, !tbaa !8
%i328B

	full_text
	
i32 %52
'i32*8B

	full_text


i32* %46
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
Ocall8BE
C
	full_text6
4
2%53 = tail call i64 @_Z14get_local_sizej(i32 0) #4
2lshr8B(
&
	full_text

%54 = lshr i64 %53, 1
%i648B

	full_text
	
i64 %53
8trunc8B-
+
	full_text

%55 = trunc i64 %54 to i32
%i648B

	full_text
	
i64 %54
5icmp8B+
)
	full_text

%56 = icmp eq i32 %55, 0
%i328B

	full_text
	
i32 %55
:br8B2
0
	full_text#
!
br i1 %56, label %58, label %57
#i18B

	full_text


i1 %56
'br8B

	full_text

br label %59
:br8B2
0
	full_text#
!
br i1 %47, label %72, label %77
#i18B

	full_text


i1 %47
Dphi8B;
9
	full_text,
*
(%60 = phi i32 [ %70, %69 ], [ %55, %57 ]
%i328B

	full_text
	
i32 %70
%i328B

	full_text
	
i32 %55
8icmp8B.
,
	full_text

%61 = icmp ugt i32 %60, %22
%i328B

	full_text
	
i32 %60
%i328B

	full_text
	
i32 %22
:br8B2
0
	full_text#
!
br i1 %61, label %62, label %69
#i18B

	full_text


i1 %61
2add8	B)
'
	full_text

%63 = add i32 %60, %22
%i328	B

	full_text
	
i32 %60
%i328	B

	full_text
	
i32 %22
6zext8	B,
*
	full_text

%64 = zext i32 %63 to i64
%i328	B

	full_text
	
i32 %63
Xgetelementptr8	BE
C
	full_text6
4
2%65 = getelementptr inbounds i32, i32* %3, i64 %64
%i648	B

	full_text
	
i64 %64
Hload8	B>
<
	full_text/
-
+%66 = load i32, i32* %65, align 4, !tbaa !8
'i32*8	B

	full_text


i32* %65
Hload8	B>
<
	full_text/
-
+%67 = load i32, i32* %46, align 4, !tbaa !8
'i32*8	B

	full_text


i32* %46
2add8	B)
'
	full_text

%68 = add i32 %67, %66
%i328	B

	full_text
	
i32 %67
%i328	B

	full_text
	
i32 %66
Hstore8	B=
;
	full_text.
,
*store i32 %68, i32* %46, align 4, !tbaa !8
%i328	B

	full_text
	
i32 %68
'i32*8	B

	full_text


i32* %46
'br8	B

	full_text

br label %69
Bcall8
B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
2lshr8
B(
&
	full_text

%70 = lshr i32 %60, 1
%i328
B

	full_text
	
i32 %60
5icmp8
B+
)
	full_text

%71 = icmp eq i32 %70, 0
%i328
B

	full_text
	
i32 %70
:br8
B2
0
	full_text#
!
br i1 %71, label %58, label %59
#i18
B

	full_text


i1 %71
Gload8B=
;
	full_text.
,
*%73 = load i32, i32* %3, align 4, !tbaa !8
1mul8B(
&
	full_text

%74 = mul i64 %9, %50
$i648B

	full_text


i64 %9
%i648B

	full_text
	
i64 %50
2add8B)
'
	full_text

%75 = add i64 %74, %13
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %13
Xgetelementptr8BE
C
	full_text6
4
2%76 = getelementptr inbounds i32, i32* %1, i64 %75
%i648B

	full_text
	
i64 %75
Hstore8B=
;
	full_text.
,
*store i32 %73, i32* %76, align 4, !tbaa !8
%i328B

	full_text
	
i32 %73
'i32*8B

	full_text


i32* %76
'br8B

	full_text

br label %77
8add8B/
-
	full_text 

%78 = add nuw nsw i64 %50, 1
%i648B

	full_text
	
i64 %50
6icmp8B,
*
	full_text

%79 = icmp eq i64 %78, 16
%i648B

	full_text
	
i64 %78
:br8B2
0
	full_text#
!
br i1 %79, label %48, label %49
#i18B

	full_text


i1 %79
&i32*8B

	full_text
	
i32* %3
$i328B

	full_text


i32 %2
&i32*8B

	full_text
	
i32* %1
$i328B

	full_text


i32 %4
&i32*8B

	full_text
	
i32* %0
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
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 64
!i88B

	full_text

i8 0
%i18B

	full_text


i1 false
$i648B

	full_text


i64 16
#i328B

	full_text	

i32 4
#i328B

	full_text	

i32 2
$i328B

	full_text


i32 31
$i648B

	full_text


i64 -1
$i328B

	full_text


i32 15
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 1        	
 		                         !" !! #$ #% ## &' && () (( *+ ** ,- ,. ,, /0 /1 22 35 46 44 78 77 9: 99 ;< ;; => =? == @A @@ BC BB DE DF DD GH GG IJ II KL KM KK NO NP NN QR QQ ST SU SS VW VY XX Z[ ZZ \] \\ ^_ ^^ `b aa ce dd fg fh ff ij ii kl km kk nn oo pq pp rs rr tu tt vw vz y| {} {{ ~ ~	Ä ~~ ÅÇ ÅÑ É
Ö ÉÉ Üá ÜÜ à
â àà äã ää åç åå éè é
ê éé ëí ë
ì ëë îï ñó ññ òô òò öõ öú ùû ù
ü ùù †° †
¢ †† £
§ ££ •¶ •
ß •• ®™ ©© ´¨ ´´ ≠Æ ≠Ø \Ø àØ ú∞ 	∞ ± £≤ 1≥ 9    
	              " $! % '& )& +# - ., 0# 5Q 64 87 :9 <; >1 ?= A@ C EB FD HG JI LD M2 O7 PN R TQ US W  YX [Z ]! _& b© e gd hf ji l\ mo qp sr ut w^ zñ |r }{ ! Ä~ Ç{ Ñ! ÖÉ áÜ âà ã\ çå èä êé í\ ì{ óñ ôò õ ûd üù ° ¢† §ú ¶£ ßd ™© ¨´ Æ/ 1/ X3 4` dV 4V Xv yv xy úy ©x {® ©≠ a≠ dÅ ÉÅ ïî ïö yö { ∑∑ c ¥¥ µµ ∂∂ ∏∏ ∫∫ ªª ππ2 ∏∏ 2  ∑∑  a ∫∫ ao ∏∏ o ∂∂  µµ ï ππ ï* ªª *( ¥¥ (n ππ nº 	º Iº nº ï
º ñΩ (	Ω *Ω a	æ *	ø *
¿ ´	¡ 	¬ 	√ 1	ƒ 	≈ @	∆ D∆ d	∆ f« « «  « 2	« ^« o	« t
« ò	» X	» Z	… p
… ©"
reduce"
llvm.lifetime.start.p0i8"
_Z14get_num_groupsj"
_Z12get_group_idj"
_Z12get_local_idj"
_Z14get_local_sizej"
_Z7barrierj"
llvm.lifetime.end.p0i8"
llvm.memset.p0i8.i64*í
shoc-1.1.5-Sort-reduce.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

wgsize_log1p
· tA

transfer_bytes
Ä†Ä

wgsize
Ä
 
transfer_bytes_log1p
· tA

devmap_label
