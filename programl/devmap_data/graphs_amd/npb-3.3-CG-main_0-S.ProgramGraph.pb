

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 0) #2
.lshrB&
$
	full_text

%6 = lshr i64 %5, 6
"i64B

	full_text


i64 %5
JcallBB
@
	full_text3
1
/%7 = tail call i64 @_Z12get_local_idj(i32 0) #2
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
-shlB&
$
	full_text

%9 = shl i64 %6, 32
"i64B

	full_text


i64 %6
6ashrB.
,
	full_text

%10 = ashr exact i64 %9, 32
"i64B

	full_text


i64 %9
VgetelementptrBE
C
	full_text6
4
2%11 = getelementptr inbounds i32, i32* %1, i64 %10
#i64B

	full_text
	
i64 %10
FloadB>
<
	full_text/
-
+%12 = load i32, i32* %11, align 4, !tbaa !9
%i32*B

	full_text


i32* %11
6addB/
-
	full_text 

%13 = add i64 %9, 4294967296
"i64B

	full_text


i64 %9
7ashrB/
-
	full_text 

%14 = ashr exact i64 %13, 32
#i64B

	full_text
	
i64 %13
VgetelementptrBE
C
	full_text6
4
2%15 = getelementptr inbounds i32, i32* %1, i64 %14
#i64B

	full_text
	
i64 %14
FloadB>
<
	full_text/
-
+%16 = load i32, i32* %15, align 4, !tbaa !9
%i32*B

	full_text


i32* %15
3addB,
*
	full_text

%17 = add nsw i32 %12, %8
#i32B

	full_text
	
i32 %12
"i32B

	full_text


i32 %8
6icmpB.
,
	full_text

%18 = icmp slt i32 %17, %16
#i32B

	full_text
	
i32 %17
#i32B

	full_text
	
i32 %16
8brB2
0
	full_text#
!
br i1 %18, label %19, label %42
!i1B

	full_text


i1 %18
6sext8B,
*
	full_text

%20 = sext i32 %17 to i64
%i328B

	full_text
	
i32 %17
6sext8B,
*
	full_text

%21 = sext i32 %16 to i64
%i328B

	full_text
	
i32 %16
5add8B,
*
	full_text

%22 = add nsw i64 %21, -1
%i648B

	full_text
	
i64 %21
6sub8B-
+
	full_text

%23 = sub nsw i64 %22, %20
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %20
2lshr8B(
&
	full_text

%24 = lshr i64 %23, 6
%i648B

	full_text
	
i64 %23
8add8B/
-
	full_text 

%25 = add nuw nsw i64 %24, 1
%i648B

	full_text
	
i64 %24
0and8B'
%
	full_text

%26 = and i64 %25, 3
%i648B

	full_text
	
i64 %25
5icmp8B+
)
	full_text

%27 = icmp eq i64 %26, 0
%i648B

	full_text
	
i64 %26
:br8B2
0
	full_text#
!
br i1 %27, label %38, label %28
#i18B

	full_text


i1 %27
'br8B

	full_text

br label %29
Dphi8B;
9
	full_text,
*
(%30 = phi i64 [ %20, %28 ], [ %35, %29 ]
%i648B

	full_text
	
i64 %20
%i648B

	full_text
	
i64 %35
Dphi8B;
9
	full_text,
*
(%31 = phi i64 [ %26, %28 ], [ %36, %29 ]
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %36
Xgetelementptr8BE
C
	full_text6
4
2%32 = getelementptr inbounds i32, i32* %0, i64 %30
%i648B

	full_text
	
i64 %30
Hload8B>
<
	full_text/
-
+%33 = load i32, i32* %32, align 4, !tbaa !9
'i32*8B

	full_text


i32* %32
5sub8B,
*
	full_text

%34 = sub nsw i32 %33, %2
%i328B

	full_text
	
i32 %33
Hstore8B=
;
	full_text.
,
*store i32 %34, i32* %32, align 4, !tbaa !9
%i328B

	full_text
	
i32 %34
'i32*8B

	full_text


i32* %32
5add8B,
*
	full_text

%35 = add nsw i64 %30, 64
%i648B

	full_text
	
i64 %30
1add8B(
&
	full_text

%36 = add i64 %31, -1
%i648B

	full_text
	
i64 %31
5icmp8B+
)
	full_text

%37 = icmp eq i64 %36, 0
%i648B

	full_text
	
i64 %36
Jbr8BB
@
	full_text3
1
/br i1 %37, label %38, label %29, !llvm.loop !13
#i18B

	full_text


i1 %37
Dphi8B;
9
	full_text,
*
(%39 = phi i64 [ %20, %19 ], [ %35, %29 ]
%i648B

	full_text
	
i64 %20
%i648B

	full_text
	
i64 %35
8icmp8B.
,
	full_text

%40 = icmp ult i64 %23, 192
%i648B

	full_text
	
i64 %23
:br8B2
0
	full_text#
!
br i1 %40, label %42, label %41
#i18B

	full_text


i1 %40
'br8B

	full_text

br label %43
$ret8B

	full_text


ret void
Dphi8B;
9
	full_text,
*
(%44 = phi i64 [ %39, %41 ], [ %60, %43 ]
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %60
Xgetelementptr8BE
C
	full_text6
4
2%45 = getelementptr inbounds i32, i32* %0, i64 %44
%i648B

	full_text
	
i64 %44
Hload8B>
<
	full_text/
-
+%46 = load i32, i32* %45, align 4, !tbaa !9
'i32*8B

	full_text


i32* %45
5sub8B,
*
	full_text

%47 = sub nsw i32 %46, %2
%i328B

	full_text
	
i32 %46
Hstore8B=
;
	full_text.
,
*store i32 %47, i32* %45, align 4, !tbaa !9
%i328B

	full_text
	
i32 %47
'i32*8B

	full_text


i32* %45
5add8B,
*
	full_text

%48 = add nsw i64 %44, 64
%i648B

	full_text
	
i64 %44
Xgetelementptr8BE
C
	full_text6
4
2%49 = getelementptr inbounds i32, i32* %0, i64 %48
%i648B

	full_text
	
i64 %48
Hload8B>
<
	full_text/
-
+%50 = load i32, i32* %49, align 4, !tbaa !9
'i32*8B

	full_text


i32* %49
5sub8B,
*
	full_text

%51 = sub nsw i32 %50, %2
%i328B

	full_text
	
i32 %50
Hstore8B=
;
	full_text.
,
*store i32 %51, i32* %49, align 4, !tbaa !9
%i328B

	full_text
	
i32 %51
'i32*8B

	full_text


i32* %49
6add8B-
+
	full_text

%52 = add nsw i64 %44, 128
%i648B

	full_text
	
i64 %44
Xgetelementptr8BE
C
	full_text6
4
2%53 = getelementptr inbounds i32, i32* %0, i64 %52
%i648B

	full_text
	
i64 %52
Hload8B>
<
	full_text/
-
+%54 = load i32, i32* %53, align 4, !tbaa !9
'i32*8B

	full_text


i32* %53
5sub8B,
*
	full_text

%55 = sub nsw i32 %54, %2
%i328B

	full_text
	
i32 %54
Hstore8B=
;
	full_text.
,
*store i32 %55, i32* %53, align 4, !tbaa !9
%i328B

	full_text
	
i32 %55
'i32*8B

	full_text


i32* %53
6add8B-
+
	full_text

%56 = add nsw i64 %44, 192
%i648B

	full_text
	
i64 %44
Xgetelementptr8BE
C
	full_text6
4
2%57 = getelementptr inbounds i32, i32* %0, i64 %56
%i648B

	full_text
	
i64 %56
Hload8B>
<
	full_text/
-
+%58 = load i32, i32* %57, align 4, !tbaa !9
'i32*8B

	full_text


i32* %57
5sub8B,
*
	full_text

%59 = sub nsw i32 %58, %2
%i328B

	full_text
	
i32 %58
Hstore8B=
;
	full_text.
,
*store i32 %59, i32* %57, align 4, !tbaa !9
%i328B

	full_text
	
i32 %59
'i32*8B

	full_text


i32* %57
6add8B-
+
	full_text

%60 = add nsw i64 %44, 256
%i648B

	full_text
	
i64 %44
8icmp8B.
,
	full_text

%61 = icmp slt i64 %60, %21
%i648B

	full_text
	
i64 %60
%i648B

	full_text
	
i64 %21
:br8B2
0
	full_text#
!
br i1 %61, label %43, label %42
#i18B

	full_text


i1 %61
&i32*8B

	full_text
	
i32* %1
&i32*8B

	full_text
	
i32* %0
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
-; undefined function B

	full_text

 
#i648B

	full_text	

i64 3
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 1
$i648B

	full_text


i64 -1
#i648B

	full_text	

i64 6
,i648B!

	full_text

i64 4294967296
%i648B

	full_text
	
i64 192
%i648B

	full_text
	
i64 128
%i648B

	full_text
	
i64 256
#i648B

	full_text	

i64 0
$i648B

	full_text


i64 64        	
 		                       !" !! #$ ## %& %' %% () (( *+ ** ,- ,, ./ .. 01 04 35 33 67 68 66 9: 99 ;< ;; => == ?@ ?A ?? BC BB DE DD FG FF HI HK JL JJ MN MM OP OT SU SS VW VV XY XX Z[ ZZ \] \^ \\ _` __ ab aa cd cc ef ee gh gi gg jk jj lm ll no nn pq pp rs rt rr uv uu wx ww yz yy {| {{ }~ } }} ?? ?? ?? ?
? ?? ?? ?? ? ? 9? V? a? l? w	? =	? Z	? e	? p	? {    
	              "! $# & '% )( +* -, /. 1 4B 5, 7D 83 :9 <; >= @9 A3 C6 ED GF I KB L% NM PJ T? US WV YX [Z ]V ^S `_ ba dc fe ha iS kj ml on qp sl tS vu xw zy |{ ~w S ?? ?! ?? ?  R0 J0 2O RO Q2 3Q SH JH 3? S? R R ?? ?? ??  ?? 	? ,? ? 	? 	? 		? 	? *	? #	? D	? 	? (	? 	? M	? u	? j
? ?	? .	? F	? B	? _"
main_0"
_Z13get_global_idj"
_Z12get_local_idj*?
npb-CG-main_0.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?
 
transfer_bytes_log1p
N?jA

devmap_label
 

wgsize
@

transfer_bytes
???

wgsize_log1p
N?jA