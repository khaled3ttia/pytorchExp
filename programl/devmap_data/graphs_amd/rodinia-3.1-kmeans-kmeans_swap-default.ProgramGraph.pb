

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 0) #2
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
%7 = icmp ult i32 %6, %2
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
 
br i1 %9, label %10, label %92
 i1B

	full_text	

i1 %9
0mul8B'
%
	full_text

%11 = mul i32 %6, %3
$i328B

	full_text


i32 %6
5zext8B+
)
	full_text

%12 = zext i32 %3 to i64
5add8B,
*
	full_text

%13 = add nsw i64 %12, -1
%i648B

	full_text
	
i64 %12
0and8B'
%
	full_text

%14 = and i64 %12, 3
%i648B

	full_text
	
i64 %12
6icmp8B,
*
	full_text

%15 = icmp ult i64 %13, 3
%i648B

	full_text
	
i64 %13
:br8B2
0
	full_text#
!
br i1 %15, label %71, label %16
#i18B

	full_text


i1 %15
6sub8B-
+
	full_text

%17 = sub nsw i64 %12, %14
%i648B

	full_text
	
i64 %12
%i648B

	full_text
	
i64 %14
'br8B

	full_text

br label %18
Bphi8B9
7
	full_text*
(
&%19 = phi i64 [ 0, %16 ], [ %68, %18 ]
%i648B

	full_text
	
i64 %68
Dphi8B;
9
	full_text,
*
(%20 = phi i64 [ %17, %16 ], [ %69, %18 ]
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %69
8trunc8B-
+
	full_text

%21 = trunc i64 %19 to i32
%i648B

	full_text
	
i64 %19
2add8B)
'
	full_text

%22 = add i32 %11, %21
%i328B

	full_text
	
i32 %11
%i328B

	full_text
	
i32 %21
6zext8B,
*
	full_text

%23 = zext i32 %22 to i64
%i328B

	full_text
	
i32 %22
\getelementptr8BI
G
	full_text:
8
6%24 = getelementptr inbounds float, float* %0, i64 %23
%i648B

	full_text
	
i64 %23
@bitcast8B3
1
	full_text$
"
 %25 = bitcast float* %24 to i32*
+float*8B

	full_text


float* %24
Hload8B>
<
	full_text/
-
+%26 = load i32, i32* %25, align 4, !tbaa !8
'i32*8B

	full_text


i32* %25
5mul8B,
*
	full_text

%27 = mul nsw i32 %21, %2
%i328B

	full_text
	
i32 %21
1add8B(
&
	full_text

%28 = add i32 %27, %6
%i328B

	full_text
	
i32 %27
$i328B

	full_text


i32 %6
6zext8B,
*
	full_text

%29 = zext i32 %28 to i64
%i328B

	full_text
	
i32 %28
\getelementptr8BI
G
	full_text:
8
6%30 = getelementptr inbounds float, float* %1, i64 %29
%i648B

	full_text
	
i64 %29
@bitcast8B3
1
	full_text$
"
 %31 = bitcast float* %30 to i32*
+float*8B

	full_text


float* %30
Hstore8B=
;
	full_text.
,
*store i32 %26, i32* %31, align 4, !tbaa !8
%i328B

	full_text
	
i32 %26
'i32*8B

	full_text


i32* %31
8trunc8B-
+
	full_text

%32 = trunc i64 %19 to i32
%i648B

	full_text
	
i64 %19
.or8B&
$
	full_text

%33 = or i32 %32, 1
%i328B

	full_text
	
i32 %32
2add8B)
'
	full_text

%34 = add i32 %11, %33
%i328B

	full_text
	
i32 %11
%i328B

	full_text
	
i32 %33
6zext8B,
*
	full_text

%35 = zext i32 %34 to i64
%i328B

	full_text
	
i32 %34
\getelementptr8BI
G
	full_text:
8
6%36 = getelementptr inbounds float, float* %0, i64 %35
%i648B

	full_text
	
i64 %35
@bitcast8B3
1
	full_text$
"
 %37 = bitcast float* %36 to i32*
+float*8B

	full_text


float* %36
Hload8B>
<
	full_text/
-
+%38 = load i32, i32* %37, align 4, !tbaa !8
'i32*8B

	full_text


i32* %37
5mul8B,
*
	full_text

%39 = mul nsw i32 %33, %2
%i328B

	full_text
	
i32 %33
1add8B(
&
	full_text

%40 = add i32 %39, %6
%i328B

	full_text
	
i32 %39
$i328B

	full_text


i32 %6
6zext8B,
*
	full_text

%41 = zext i32 %40 to i64
%i328B

	full_text
	
i32 %40
\getelementptr8BI
G
	full_text:
8
6%42 = getelementptr inbounds float, float* %1, i64 %41
%i648B

	full_text
	
i64 %41
@bitcast8B3
1
	full_text$
"
 %43 = bitcast float* %42 to i32*
+float*8B

	full_text


float* %42
Hstore8B=
;
	full_text.
,
*store i32 %38, i32* %43, align 4, !tbaa !8
%i328B

	full_text
	
i32 %38
'i32*8B

	full_text


i32* %43
8trunc8B-
+
	full_text

%44 = trunc i64 %19 to i32
%i648B

	full_text
	
i64 %19
.or8B&
$
	full_text

%45 = or i32 %44, 2
%i328B

	full_text
	
i32 %44
2add8B)
'
	full_text

%46 = add i32 %11, %45
%i328B

	full_text
	
i32 %11
%i328B

	full_text
	
i32 %45
6zext8B,
*
	full_text

%47 = zext i32 %46 to i64
%i328B

	full_text
	
i32 %46
\getelementptr8BI
G
	full_text:
8
6%48 = getelementptr inbounds float, float* %0, i64 %47
%i648B

	full_text
	
i64 %47
@bitcast8B3
1
	full_text$
"
 %49 = bitcast float* %48 to i32*
+float*8B

	full_text


float* %48
Hload8B>
<
	full_text/
-
+%50 = load i32, i32* %49, align 4, !tbaa !8
'i32*8B

	full_text


i32* %49
5mul8B,
*
	full_text

%51 = mul nsw i32 %45, %2
%i328B

	full_text
	
i32 %45
1add8B(
&
	full_text

%52 = add i32 %51, %6
%i328B

	full_text
	
i32 %51
$i328B

	full_text


i32 %6
6zext8B,
*
	full_text

%53 = zext i32 %52 to i64
%i328B

	full_text
	
i32 %52
\getelementptr8BI
G
	full_text:
8
6%54 = getelementptr inbounds float, float* %1, i64 %53
%i648B

	full_text
	
i64 %53
@bitcast8B3
1
	full_text$
"
 %55 = bitcast float* %54 to i32*
+float*8B

	full_text


float* %54
Hstore8B=
;
	full_text.
,
*store i32 %50, i32* %55, align 4, !tbaa !8
%i328B

	full_text
	
i32 %50
'i32*8B

	full_text


i32* %55
8trunc8B-
+
	full_text

%56 = trunc i64 %19 to i32
%i648B

	full_text
	
i64 %19
.or8B&
$
	full_text

%57 = or i32 %56, 3
%i328B

	full_text
	
i32 %56
2add8B)
'
	full_text

%58 = add i32 %11, %57
%i328B

	full_text
	
i32 %11
%i328B

	full_text
	
i32 %57
6zext8B,
*
	full_text

%59 = zext i32 %58 to i64
%i328B

	full_text
	
i32 %58
\getelementptr8BI
G
	full_text:
8
6%60 = getelementptr inbounds float, float* %0, i64 %59
%i648B

	full_text
	
i64 %59
@bitcast8B3
1
	full_text$
"
 %61 = bitcast float* %60 to i32*
+float*8B

	full_text


float* %60
Hload8B>
<
	full_text/
-
+%62 = load i32, i32* %61, align 4, !tbaa !8
'i32*8B

	full_text


i32* %61
5mul8B,
*
	full_text

%63 = mul nsw i32 %57, %2
%i328B

	full_text
	
i32 %57
1add8B(
&
	full_text

%64 = add i32 %63, %6
%i328B

	full_text
	
i32 %63
$i328B

	full_text


i32 %6
6zext8B,
*
	full_text

%65 = zext i32 %64 to i64
%i328B

	full_text
	
i32 %64
\getelementptr8BI
G
	full_text:
8
6%66 = getelementptr inbounds float, float* %1, i64 %65
%i648B

	full_text
	
i64 %65
@bitcast8B3
1
	full_text$
"
 %67 = bitcast float* %66 to i32*
+float*8B

	full_text


float* %66
Hstore8B=
;
	full_text.
,
*store i32 %62, i32* %67, align 4, !tbaa !8
%i328B

	full_text
	
i32 %62
'i32*8B

	full_text


i32* %67
4add8B+
)
	full_text

%68 = add nsw i64 %19, 4
%i648B

	full_text
	
i64 %19
1add8B(
&
	full_text

%69 = add i64 %20, -4
%i648B

	full_text
	
i64 %20
5icmp8B+
)
	full_text

%70 = icmp eq i64 %69, 0
%i648B

	full_text
	
i64 %69
:br8B2
0
	full_text#
!
br i1 %70, label %71, label %18
#i18B

	full_text


i1 %70
Bphi8B9
7
	full_text*
(
&%72 = phi i64 [ 0, %10 ], [ %68, %18 ]
%i648B

	full_text
	
i64 %68
5icmp8B+
)
	full_text

%73 = icmp eq i64 %14, 0
%i648B

	full_text
	
i64 %14
:br8B2
0
	full_text#
!
br i1 %73, label %92, label %74
#i18B

	full_text


i1 %73
'br8B

	full_text

br label %75
Dphi8B;
9
	full_text,
*
(%76 = phi i64 [ %72, %74 ], [ %89, %75 ]
%i648B

	full_text
	
i64 %72
%i648B

	full_text
	
i64 %89
Dphi8B;
9
	full_text,
*
(%77 = phi i64 [ %14, %74 ], [ %90, %75 ]
%i648B

	full_text
	
i64 %14
%i648B

	full_text
	
i64 %90
8trunc8B-
+
	full_text

%78 = trunc i64 %76 to i32
%i648B

	full_text
	
i64 %76
2add8B)
'
	full_text

%79 = add i32 %11, %78
%i328B

	full_text
	
i32 %11
%i328B

	full_text
	
i32 %78
6zext8B,
*
	full_text

%80 = zext i32 %79 to i64
%i328B

	full_text
	
i32 %79
\getelementptr8BI
G
	full_text:
8
6%81 = getelementptr inbounds float, float* %0, i64 %80
%i648B

	full_text
	
i64 %80
@bitcast8B3
1
	full_text$
"
 %82 = bitcast float* %81 to i32*
+float*8B

	full_text


float* %81
Hload8B>
<
	full_text/
-
+%83 = load i32, i32* %82, align 4, !tbaa !8
'i32*8B

	full_text


i32* %82
5mul8B,
*
	full_text

%84 = mul nsw i32 %78, %2
%i328B

	full_text
	
i32 %78
1add8B(
&
	full_text

%85 = add i32 %84, %6
%i328B

	full_text
	
i32 %84
$i328B

	full_text


i32 %6
6zext8B,
*
	full_text

%86 = zext i32 %85 to i64
%i328B

	full_text
	
i32 %85
\getelementptr8BI
G
	full_text:
8
6%87 = getelementptr inbounds float, float* %1, i64 %86
%i648B

	full_text
	
i64 %86
@bitcast8B3
1
	full_text$
"
 %88 = bitcast float* %87 to i32*
+float*8B

	full_text


float* %87
Hstore8B=
;
	full_text.
,
*store i32 %83, i32* %88, align 4, !tbaa !8
%i328B

	full_text
	
i32 %83
'i32*8B

	full_text


i32* %88
8add8B/
-
	full_text 

%89 = add nuw nsw i64 %76, 1
%i648B

	full_text
	
i64 %76
1add8B(
&
	full_text

%90 = add i64 %77, -1
%i648B

	full_text
	
i64 %77
5icmp8B+
)
	full_text

%91 = icmp eq i64 %90, 0
%i648B

	full_text
	
i64 %90
Jbr8BB
@
	full_text3
1
/br i1 %91, label %92, label %75, !llvm.loop !12
#i18B

	full_text


i1 %91
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %1
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %2
-; undefined function B

	full_text

 
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 3
#i648B

	full_text	

i64 4
#i328B

	full_text	

i32 2
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 3
$i648B

	full_text


i64 -4
#i648B

	full_text	

i64 1
$i648B

	full_text


i64 -1       	  
 
                   !    "# "$ "" %& %% '( '' )* )) +, ++ -. -- /0 /1 // 23 22 45 44 67 66 89 8: 88 ;< ;; => == ?@ ?A ?? BC BB DE DD FG FF HI HH JK JJ LM LN LL OP OO QR QQ ST SS UV UW UU XY XX Z[ ZZ \] \^ \\ _` __ ab aa cd cc ef ee gh gg ij ik ii lm ll no nn pq pp rs rt rr uv uu wx ww yz y{ yy |} || ~ ~~ ÄÅ ÄÄ ÇÉ ÇÇ ÑÖ ÑÑ Üá Ü
à ÜÜ âä ââ ã
å ãã çé çç èê è
ë èè íì íí îï îî ñó ññ òô ò
õ öö úù úú ûü û¢ °
£ °° §• §
¶ §§ ß® ßß ©™ ©
´ ©© ¨≠ ¨¨ Æ
Ø ÆÆ ∞± ∞∞ ≤≥ ≤≤ ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ ππ ª
º ªª Ωæ ΩΩ ø¿ ø
¡ øø ¬√ ¬¬ ƒ≈ ƒƒ ∆« ∆∆ »… »À 4À QÀ nÀ ãÀ ªÃ 	Ã Ã Õ 'Õ DÕ aÕ ~Õ Æ	Œ 	Œ -	Œ J	Œ g
Œ Ñ
Œ ¥    	        í  î  ! #  $" &% (' *) ,  .- 0 1/ 32 54 7+ 96 : <; > @= A? CB ED GF I= KJ M NL PO RQ TH VS W YX [ ]Z ^\ `_ ba dc fZ hg j ki ml on qe sp t vu x zw {y }| ~ ÅÄ Éw ÖÑ á àÜ äâ åã éÇ êç ë ì ïî óñ ôí õ ùú üö ¢¬ £ •ƒ ¶° ® ™ß ´© ≠¨ ØÆ ±∞ ≥ß µ¥ ∑ ∏∂ ∫π ºª æ≤ ¿Ω ¡° √§ ≈ƒ «∆ …
 
   ö û  û † † °ò öò »  » ° œœ   œœ – 
– ñ– ö
– ú
– ∆	— =	“ w
” í	‘ Z’ 	’ 	÷ 	÷ 
◊ î
ÿ ¬	Ÿ 
Ÿ ƒ"
kmeans_swap"
_Z13get_global_idj*ö
!rodinia-3.1-kmeans-kmeans_swap.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

wgsize_log1p
çÿïA

devmap_label


transfer_bytes
¯ìÇA

wgsize
Ä
 
transfer_bytes_log1p
çÿïA