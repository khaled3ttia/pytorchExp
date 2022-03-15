

[external]
JcallBB
@
	full_text3
1
/%4 = tail call i64 @_Z12get_group_idj(i32 0) #2
4truncB+
)
	full_text

%5 = trunc i64 %4 to i32
"i64B

	full_text


i64 %4
JcallBB
@
	full_text3
1
/%6 = tail call i64 @_Z12get_group_idj(i32 1) #2
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
JcallBB
@
	full_text3
1
/%8 = tail call i64 @_Z12get_local_idj(i32 0) #2
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
McallBE
C
	full_text6
4
2%10 = tail call i32 @_Z5mul24ii(i32 %1, i32 %2) #2
5mulB.
,
	full_text

%11 = mul nsw i32 %10, 1096
#i32B

	full_text
	
i32 %10
McallBE
C
	full_text6
4
2%12 = tail call i32 @_Z5mul24ii(i32 %7, i32 %1) #2
"i32B

	full_text


i32 %7
3addB,
*
	full_text

%13 = add nsw i32 %12, %5
#i32B

	full_text
	
i32 %12
"i32B

	full_text


i32 %5
5mulB.
,
	full_text

%14 = mul nsw i32 %13, 1096
#i32B

	full_text
	
i32 %13
4addB-
+
	full_text

%15 = add nsw i32 %14, %11
#i32B

	full_text
	
i32 %14
#i32B

	full_text
	
i32 %11
4sextB,
*
	full_text

%16 = sext i32 %11 to i64
#i32B

	full_text
	
i32 %11
5icmpB-
+
	full_text

%17 = icmp slt i32 %9, 545
"i32B

	full_text


i32 %9
8brB2
0
	full_text#
!
br i1 %17, label %18, label %64
!i1B

	full_text


i1 %17
6sext8B,
*
	full_text

%19 = sext i32 %14 to i64
%i328B

	full_text
	
i32 %14
Xgetelementptr8BE
C
	full_text6
4
2%20 = getelementptr inbounds i16, i16* %0, i64 %19
%i648B

	full_text
	
i64 %19
Xgetelementptr8BE
C
	full_text6
4
2%21 = getelementptr inbounds i16, i16* %0, i64 %16
%i648B

	full_text
	
i64 %16
3mul8B*
(
	full_text

%22 = mul i32 %13, 2192
%i328B

	full_text
	
i32 %13
6sext8B,
*
	full_text

%23 = sext i32 %22 to i64
%i328B

	full_text
	
i32 %22
Ygetelementptr8BF
D
	full_text7
5
3%24 = getelementptr inbounds i16, i16* %21, i64 %23
'i16*8B

	full_text


i16* %21
%i648B

	full_text
	
i64 %23
0shl8B'
%
	full_text

%25 = shl i32 %15, 1
%i328B

	full_text
	
i32 %15
6sext8B,
*
	full_text

%26 = sext i32 %25 to i64
%i328B

	full_text
	
i32 %25
Xgetelementptr8BE
C
	full_text6
4
2%27 = getelementptr inbounds i16, i16* %0, i64 %26
%i648B

	full_text
	
i64 %26
Ygetelementptr8BF
D
	full_text7
5
3%28 = getelementptr inbounds i16, i16* %27, i64 %16
'i16*8B

	full_text


i16* %27
%i648B

	full_text
	
i64 %16
0shl8B'
%
	full_text

%29 = shl i32 %15, 2
%i328B

	full_text
	
i32 %15
6sext8B,
*
	full_text

%30 = sext i32 %29 to i64
%i328B

	full_text
	
i32 %29
Xgetelementptr8BE
C
	full_text6
4
2%31 = getelementptr inbounds i16, i16* %0, i64 %30
%i648B

	full_text
	
i64 %30
Ygetelementptr8BF
D
	full_text7
5
3%32 = getelementptr inbounds i16, i16* %31, i64 %16
'i16*8B

	full_text


i16* %31
%i648B

	full_text
	
i64 %16
>bitcast8B1
/
	full_text"
 
%33 = bitcast i16* %32 to i32*
'i16*8B

	full_text


i16* %32
>bitcast8B1
/
	full_text"
 
%34 = bitcast i16* %28 to i32*
'i16*8B

	full_text


i16* %28
>bitcast8B1
/
	full_text"
 
%35 = bitcast i16* %24 to i32*
'i16*8B

	full_text


i16* %24
>bitcast8B1
/
	full_text"
 
%36 = bitcast i16* %20 to i32*
'i16*8B

	full_text


i16* %20
0shl8B'
%
	full_text

%37 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%38 = ashr exact i64 %37, 32
%i648B

	full_text
	
i64 %37
'br8B

	full_text

br label %39
Dphi8B;
9
	full_text,
*
(%40 = phi i64 [ %38, %18 ], [ %62, %39 ]
%i648B

	full_text
	
i64 %38
%i648B

	full_text
	
i64 %62
Ygetelementptr8BF
D
	full_text7
5
3%41 = getelementptr inbounds i32, i32* %33, i64 %40
'i32*8B

	full_text


i32* %33
%i648B

	full_text
	
i64 %40
Hload8B>
<
	full_text/
-
+%42 = load i32, i32* %41, align 4, !tbaa !8
'i32*8B

	full_text


i32* %41
6add8B-
+
	full_text

%43 = add nsw i64 %40, 548
%i648B

	full_text
	
i64 %40
Ygetelementptr8BF
D
	full_text7
5
3%44 = getelementptr inbounds i32, i32* %33, i64 %43
'i32*8B

	full_text


i32* %33
%i648B

	full_text
	
i64 %43
Hload8B>
<
	full_text/
-
+%45 = load i32, i32* %44, align 4, !tbaa !8
'i32*8B

	full_text


i32* %44
7add8B.
,
	full_text

%46 = add nsw i64 %40, 1096
%i648B

	full_text
	
i64 %40
Ygetelementptr8BF
D
	full_text7
5
3%47 = getelementptr inbounds i32, i32* %33, i64 %46
'i32*8B

	full_text


i32* %33
%i648B

	full_text
	
i64 %46
Hload8B>
<
	full_text/
-
+%48 = load i32, i32* %47, align 4, !tbaa !8
'i32*8B

	full_text


i32* %47
7add8B.
,
	full_text

%49 = add nsw i64 %40, 1644
%i648B

	full_text
	
i64 %40
Ygetelementptr8BF
D
	full_text7
5
3%50 = getelementptr inbounds i32, i32* %33, i64 %49
'i32*8B

	full_text


i32* %33
%i648B

	full_text
	
i64 %49
Hload8B>
<
	full_text/
-
+%51 = load i32, i32* %50, align 4, !tbaa !8
'i32*8B

	full_text


i32* %50
2add8B)
'
	full_text

%52 = add i32 %48, %42
%i328B

	full_text
	
i32 %48
%i328B

	full_text
	
i32 %42
Ygetelementptr8BF
D
	full_text7
5
3%53 = getelementptr inbounds i32, i32* %34, i64 %40
'i32*8B

	full_text


i32* %34
%i648B

	full_text
	
i64 %40
Hstore8B=
;
	full_text.
,
*store i32 %52, i32* %53, align 4, !tbaa !8
%i328B

	full_text
	
i32 %52
'i32*8B

	full_text


i32* %53
2add8B)
'
	full_text

%54 = add i32 %51, %45
%i328B

	full_text
	
i32 %51
%i328B

	full_text
	
i32 %45
Ygetelementptr8BF
D
	full_text7
5
3%55 = getelementptr inbounds i32, i32* %34, i64 %43
'i32*8B

	full_text


i32* %34
%i648B

	full_text
	
i64 %43
Hstore8B=
;
	full_text.
,
*store i32 %54, i32* %55, align 4, !tbaa !8
%i328B

	full_text
	
i32 %54
'i32*8B

	full_text


i32* %55
2add8B)
'
	full_text

%56 = add i32 %45, %42
%i328B

	full_text
	
i32 %45
%i328B

	full_text
	
i32 %42
Ygetelementptr8BF
D
	full_text7
5
3%57 = getelementptr inbounds i32, i32* %35, i64 %40
'i32*8B

	full_text


i32* %35
%i648B

	full_text
	
i64 %40
Hstore8B=
;
	full_text.
,
*store i32 %56, i32* %57, align 4, !tbaa !8
%i328B

	full_text
	
i32 %56
'i32*8B

	full_text


i32* %57
2add8B)
'
	full_text

%58 = add i32 %51, %48
%i328B

	full_text
	
i32 %51
%i328B

	full_text
	
i32 %48
Ygetelementptr8BF
D
	full_text7
5
3%59 = getelementptr inbounds i32, i32* %35, i64 %43
'i32*8B

	full_text


i32* %35
%i648B

	full_text
	
i64 %43
Hstore8B=
;
	full_text.
,
*store i32 %58, i32* %59, align 4, !tbaa !8
%i328B

	full_text
	
i32 %58
'i32*8B

	full_text


i32* %59
2add8B)
'
	full_text

%60 = add i32 %58, %56
%i328B

	full_text
	
i32 %58
%i328B

	full_text
	
i32 %56
Ygetelementptr8BF
D
	full_text7
5
3%61 = getelementptr inbounds i32, i32* %36, i64 %40
'i32*8B

	full_text


i32* %36
%i648B

	full_text
	
i64 %40
Hstore8B=
;
	full_text.
,
*store i32 %60, i32* %61, align 4, !tbaa !8
%i328B

	full_text
	
i32 %60
'i32*8B

	full_text


i32* %61
5add8B,
*
	full_text

%62 = add nsw i64 %40, 32
%i648B

	full_text
	
i64 %40
8icmp8B.
,
	full_text

%63 = icmp slt i64 %40, 513
%i648B

	full_text
	
i64 %40
:br8B2
0
	full_text#
!
br i1 %63, label %39, label %64
#i18B

	full_text


i1 %63
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %1
$i328B

	full_text


i32 %2
&i16*8B

	full_text
	
i16* %0
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
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 2
&i328B

	full_text


i32 1096
&i328B

	full_text


i32 2192
$i648B

	full_text


i64 32
%i648B

	full_text
	
i64 548
&i648B

	full_text


i64 1644
#i328B

	full_text	

i32 1
%i328B

	full_text
	
i32 545
&i648B

	full_text


i64 1096
%i648B

	full_text
	
i64 513       	  

                       !" !! #$ ## %& %% '( ') '' *+ ** ,- ,, ./ .. 01 02 00 34 33 56 55 78 77 9: 9; 99 <= << >? >> @A @@ BC BB DE DD FG FF HJ IK II LM LN LL OP OO QR QQ ST SU SS VW VV XY XX Z[ Z\ ZZ ]^ ]] _` __ ab ac aa de dd fg fh ff ij ik ii lm ln ll op oq oo rs rt rr uv uw uu xy xz xx {| {} {{ ~ ~	Ä ~~ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü ÑÑ áà á
â áá äã ä
å ää çé ç
è çç êë ê
í êê ìî ìì ïñ ïï óò óö 
	ö 	õ 
ú ú !ú .ú 7   	
              " $# &! (% ) +* -, /. 1 2 43 65 87 : ;9 =0 ?' A C ED GF Jì K< MI NL PI R< TQ US WI Y< [X \Z ^I `< b_ ca e] gO h> jI kf mi nd pV q> sQ to vr wV yO z@ |I }x { Äd Ç] É@ ÖQ ÜÅ àÑ âÅ ãx åB éI èä ëç íI îI ñï ò  ôH Ió Ió ô ùù ô üü ûû
 üü 
 ùù  ûû  üü  ùù † † 	° 3	¢ 	¢ 	£ #	§ D	§ F
§ ì	• Q	¶ _ß 	ß *	® 	© X
™ ï"
larger_sad_calc_16"
_Z12get_group_idj"
_Z12get_local_idj"

_Z5mul24ii*û
%parboil-0.2-sad-larger_sad_calc_16.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Ä
 
transfer_bytes_log1p
˙ìÖA

wgsize_log1p
˙ìÖA

transfer_bytes
‡ò¡

devmap_label
 

wgsize
 